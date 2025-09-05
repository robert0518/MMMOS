import os
import sys
import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import torchaudio
import random
# import your model; adjust this path if needed
from dual_crit_qual_loss import DualCriterionQualityLoss
from concordance_corr_coef_loss import ConcordanceCorrCoefLoss
from contrastive_loss import ClippedMSEContrastiveLoss, ContrastiveOnlyLoss
from transformers import WavLMModel
from aes_M2d import AesMultiOutput

# Logging setup: send INFO to stderr
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

WAVLM_CKPT_PATH = "your/path/to/WavLM-Base+.pt"


class AudioAestheticsDataset(Dataset):
    """
    Dataset for audio aesthetic scoring.
    - Reads CSV, builds full paths, filters out missing files
    - Resamples audio to 16 kHz, converts to mono
    - Randomly crops 10 s chunks if longer, pads if shorter
    - Z-normalizes targets to zero mean, unit variance
    """
    def __init__(self, csv_path, data_root, normalize_targets=True):
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root

        # Build full file paths
        self.df['full_path'] = self.df['data_path'].apply(
            lambda p: p.replace('/your_path', self.data_root)
        )
        # Filter out missing files
        exists_mask = self.df['full_path'].apply(os.path.exists)
        missing_count = (~exists_mask).sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing audio files; these will be skipped.")
        self.df = self.df[exists_mask].reset_index(drop=True)

        # Pre-compute target means & stds for z-normalization as float32 tensors
        if normalize_targets:
            cols = ["Content_Enjoyment", "Content_Usefulness", "Production_Complexity", "Production_Quality"]
            means = self.df[cols].mean().values.astype(np.float32)
            stds  = self.df[cols].std().values.astype(np.float32)
            self.target_means = torch.from_numpy(means)
            self.target_stds  = torch.from_numpy(stds)
        else:
            self.target_means = None

        self.chunk_size = 16000 * 10  # 10 seconds @16kHz

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['full_path']

        # Load audio
        try:
            wav, sr = torchaudio.load(path)
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            wav = torch.zeros(1, self.chunk_size)
            sr = 16000

        # Resample if needed
        if sr != 16000:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wav)

        # Convert to mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Crop or pad to exactly 10 s
        total_len = wav.size(1)
        if total_len > self.chunk_size:
            start = torch.randint(0, total_len - self.chunk_size + 1, (1,)).item()
            wav = wav[:, start:start + self.chunk_size]
        elif total_len < self.chunk_size:
            pad_len = self.chunk_size - total_len
            wav = F.pad(wav, (0, pad_len))

        target = torch.tensor([
            row["Content_Enjoyment"],
            row["Content_Usefulness"],
            row["Production_Complexity"],
            row["Production_Quality"],
        ], dtype=torch.float32)

        if self.target_means is not None:
            target = (target - self.target_means) / self.target_stds

        return {"wav": wav, "target": target}


def parse_args():
    p = argparse.ArgumentParser(description="Train Audio Aesthetics Model")
    p.add_argument(
        "--train_csv",
        default="your/path/to/audiomos2025_track2/combined_train.csv",
        help="Path to the training CSV list"
    )
    p.add_argument(
        "--dev_csv",
        default="your/path/to/audiomos2025_track2/combined_dev.csv",
        help="Path to the development/validation CSV list"
    )
    p.add_argument(
        "--train_root",
        default="your/path/to",
        help="Root directory for train audio files"
    )
    p.add_argument(
        "--dev_root",
        default="your/path/to",
        help="Root directory for dev audio files"
    )
    p.add_argument("--loss_fn_id", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    p.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    p.add_argument("--gpu", default="0", help='GPU device IDs, e.g. "0,1"; use "-1" for CPU')
    p.add_argument("--seed", default=42)
    return p.parse_args()

def confirm_loss_function(loss_fn_id):
    if loss_fn_id == 1:
        loss_fn = ClippedMSEContrastiveLoss()
    elif loss_fn_id == 2:
        loss_fn = ContrastiveOnlyLoss()
    elif loss_fn_id == 3:
        loss_fn = DualCriterionQualityLoss()
    elif loss_fn_id == 4:
        loss_fn = ConcordanceCorrCoefLoss()

    return loss_fn
def load_aes_with_wavlm_checkpoint(device: torch.device, **aes_kwargs) -> AesMultiOutput:
    logger.info(f"Loading WavLM checkpoint from {WAVLM_CKPT_PATH}")
    ckpt = torch.load(WAVLM_CKPT_PATH, map_location=device)
    state_dict = ckpt.get("model", ckpt)

    model = AesMultiOutput(**aes_kwargs).to(device)
    load_res = model.wavlm_model.load_state_dict(state_dict, strict=False)
    logger.info(f"Backbone missing keys: {load_res.missing_keys}")
    logger.info(f"Backbone unexpected keys: {load_res.unexpected_keys}")
    return model


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")

    global_step = 0

    # ---- data ----
    train_ds = AudioAestheticsDataset(args.train_csv, args.train_root, normalize_targets=True)
    val_ds   = AudioAestheticsDataset(args.dev_csv,   args.dev_root,   normalize_targets=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ---- model (freeze encoder!) ----
    model = load_aes_with_wavlm_checkpoint(
        device=device
    )
    model.to(device)

    if torch.cuda.device_count() > 1 and "," in args.gpu:
        model = nn.DataParallel(model)
        logger.info(f"DataParallel on GPUs {args.gpu}")
    head = model.module if isinstance(model, nn.DataParallel) else model

    # # ---- break symmetry: init each head bias to 0.1 ----
    # for axis in ["CE", "CU", "PC", "PQ"]:
    #     head.proj_layer[axis][-1].bias.data.fill_(0.1)

    # ---- optimizer (fixed LR, no scheduler) ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = float("inf")
    print(args.loss_fn_id)
    loss_fn = confirm_loss_function(args.loss_fn_id)
    
    for epoch in range(1, args.epochs + 1):
        # — train —
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            wavs    = batch["wav"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            preds      = model({"wav": wavs})
            pred_stack = torch.stack([preds[a] for a in ["CE", "CU", "PC", "PQ"]], dim=1)
            # Forward
            optimizer.zero_grad()
            preds_dict = model({"wav": wavs})
            # Stack preds into [B, 4] in the order ["CE", "CU", "PC", "PQ"]
            pred_stack = torch.stack([preds_dict[a] for a in ["CE", "CU", "PC", "PQ"]], dim=1)
            # Compute CCC loss for each axis and average
            loss = 0.0
            for i in range(4):
                # Each call: pred_stack[:, i] vs. targets[:, i]
                loss += loss_fn(pred_stack[:, i], targets[:, i])

            loss = loss / 4.0

            # Backprop
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # log epoch-level train loss
        avg_trn = sum(train_losses) / len(train_losses)
        
        # — validate ConLoss—
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                wavs    = batch["wav"].to(device)
                targets = batch["target"].to(device)
                preds   = model({"wav": wavs})
                pred_stack = torch.stack([preds[a] for a in ["CE", "CU", "PC", "PQ"]], dim=1)  # [B,4]

                loss = 0.0
                for i in range(4):
                    # Each call: pred_stack[:, i] vs. targets[:, i]
                    loss += loss_fn(pred_stack[:, i], targets[:, i])

                loss = loss / 4.0
                val_losses.append(loss.item())

        avg_val = sum(val_losses) / len(val_losses)
        logger.info(f"Epoch {epoch} → train {avg_trn:.4f}, val {avg_val:.4f}")

    # checkpoint best
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), "your/path/to/checkpoint")
            logger.info("  → new best model saved")


if __name__ == "__main__":
    main()
