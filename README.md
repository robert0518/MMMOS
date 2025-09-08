# MMMOS: Multi-domain Multi-axis Audio Quality Assessment

📄 **Presented at IEEE ASRU 2025**  
📚 **Paper**: [Arxiv 2507.04094](https://arxiv.org/abs/2507.04094)

---

## 📂 Dataset

The training and validation datasets used in this work are located in the `audiomos2025_track2` directory.

| Dataset | URL |
|---------|-----|
| LibriTTS | [https://openslr.org/60/](https://openslr.org/60/) |
| CommonVoice (cv-corpus-13.0-2023-03-09) | [https://commonvoice.mozilla.org/en/datasets](https://commonvoice.mozilla.org/en/datasets) |
| EARS | [https://sp-uhh.github.io/ears_dataset/](https://sp-uhh.github.io/ears_dataset/) |
| MUSDB18 | [https://sigsep.github.io/datasets/musdb.html](https://sigsep.github.io/datasets/musdb.html) |
| MusicCaps | [https://www.kaggle.com/datasets/googleai/musiccaps](https://www.kaggle.com/datasets/googleai/musiccaps) |
| AudioSet (unbalanced_train_segments) | [https://research.google.com/audioset/dataset/index.html](https://research.google.com/audioset/dataset/index.html) |
| PAM | [https://zenodo.org/records/10737388](https://zenodo.org/records/10737388) |

---

## 🧪 Experiments

### Prepare checkpoints
Update paths in your environment and load **WavLM** and **M2d** checkpoints.

- **`aes_M2d.py`**
  | Path | Description |
  |------|-------------|
  | `m2d_ckpt` | Path to your M2d checkpoint |

- **`train.py`**
  | Path | Description |
  |------|-------------|
  | `WAVLM_CKPT_PATH` | Path to WavLM checkpoint|
  | `train_csv` | Path to training CSV file |
  | `dev_csv` | Path to validation CSV file |
  | `train_root` | Root path to training dataset |
  | `dev_root` | Root path to validation dataset |

---

### Configure loss function
Select the loss function using the `--loss_fn_id` argument:

| ID | Loss Function |
|----|---------------|
| `1` | UTMOS loss |
| `2` | Contrastive loss |
| `3` | Dual Criterion Quality loss |
| `4` | Concordance Correlation Coefficient (CCC) loss |

---

### Training arguments
The script `train.py` accepts the following arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--loss_fn_id` | Loss function ID (see above) | *required* |
| `--lr` | Initial learning rate | `1e-4` |
| `--batch_size` | Batch size | `4` |
| `--epochs` | Number of training epochs | `10` |
| `--gpu` | GPU device ID | `0` |
| `--seed` | Random seed | `42` |

---

### Run training
Example command:

```bash
python train.py \
  --loss_fn_id 1 \
  --gpu 0 \
  --lr 1e-4 \
  --batch_size 16 \
  --epochs 10
