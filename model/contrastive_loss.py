import torch
import torch.nn as nn
import torch.nn.functional as F


class ClippedMSEContrastiveLoss(nn.Module):
    """
    Implements the loss:
        L = L_reg + w_con * L_con
    where
      - L_reg is a clipped-MSE with threshold tau
      - L_con is the pairwise contrastive term with margin

    Args:
        tau (float): clipping threshold for L_reg (default: 0.5)
        margin (float): margin for L_con (default: 0.1)
        w_con (float): weight on contrastive term (default: 0.5)
    """
    def __init__(self, tau: float = 0.5, margin: float = 0.1, w_con: float = 0.5):
        super().__init__()
        self.tau = tau
        self.margin = margin
        self.w_con = w_con

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_score: [B, A] (e.g., A=4 for CE, CU, PC, PQ) or [B] / [B,1]
            gt_score:   [B, A] to match pred, or [B] / [B,1]
        Returns:
            Scalar loss tensor.
        """
        pred, gt = _ensure_BA(pred_score, gt_score)  # [B, A], [B, A]
        B, A = pred.shape

        # 1) clipped-MSE regression loss (tau)
        abs_err = (pred - gt).abs()                          # [B, A]
        mask = (abs_err > self.tau).float()                  # [B, A]
        L_reg = (mask * (pred - gt).pow(2)).mean()           # scalar

        # 2) pairwise contrastive loss (margin)
        if B > 1:
            true_diff = gt.unsqueeze(1) - gt.unsqueeze(0)    # [B, B, A]
            pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
            con_mat = (true_diff - pred_diff).abs() - self.margin
            con_mat = F.relu(con_mat)

            # zero out i==j
            eye = torch.eye(B, device=con_mat.device, dtype=con_mat.dtype).unsqueeze(-1)
            con_mat = con_mat * (1 - eye)

            L_con = con_mat.sum() / (B * (B - 1) * A)
        else:
            # No pairs exist if B==1; treat contrastive term as 0
            L_con = torch.zeros((), device=pred.device, dtype=pred.dtype)

        return L_reg + self.w_con * L_con


class ContrastiveOnlyLoss(nn.Module):
    """
    Implements:
        L = L_con
    with the same pairwise contrastive term and margin.
    """
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor) -> torch.Tensor:
        pred, gt = _ensure_BA(pred_score, gt_score)  # [B, A], [B, A]
        B, A = pred.shape

        if B > 1:
            true_diff = gt.unsqueeze(1) - gt.unsqueeze(0)    # [B, B, A]
            pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
            con_mat = (true_diff - pred_diff).abs() - self.margin
            con_mat = F.relu(con_mat)

            eye = torch.eye(B, device=con_mat.device, dtype=con_mat.dtype).unsqueeze(-1)
            con_mat = con_mat * (1 - eye)

            L_con = con_mat.sum() / (B * (B - 1) * A)
        else:
            L_con = torch.zeros((), device=pred.device, dtype=pred.dtype)

        return L_con


def _ensure_BA(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize shapes to [B, A] for both pred and gt.
    - Accepts [B], [B,1], or [B,A].
    - If tensors have >2 dims (e.g., token/time dim), average across dim=1.
    """
    # Handle pred
    if pred.dim() > 2:
        pred = pred.mean(1)
    if pred.dim() == 1:
        pred = pred.unsqueeze(1)  # [B] -> [B,1]
    elif pred.dim() != 2:
        raise ValueError(f"pred_score must be [B], [B,1], or [B,A]; got {tuple(pred.shape)}")

    # Handle gt
    if gt.dim() > 2:
        gt = gt.mean(1)
    if gt.dim() == 1:
        gt = gt.unsqueeze(1)      # [B] -> [B,1]
    elif gt.dim() != 2:
        raise ValueError(f"gt_score must be [B], [B,1], or [B,A]; got {tuple(gt.shape)}")

    # Broadcast gt to match pred's A if needed (e.g., gt [B,1], pred [B,A])
    if gt.shape[1] == 1 and pred.shape[1] > 1:
        gt = gt.expand(-1, pred.shape[1])
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt shapes must match after normalization; got {tuple(pred.shape)} vs {tuple(gt.shape)}")

    return pred, gt
