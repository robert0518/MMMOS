import torch
import torch.nn as nn

class DualCriterionQualityLoss(nn.Module):
    """
    DCQ Loss: Combines MSE loss with Relative Perception Constraint (RPC)
    which includes:
    - QDC: Quantitative Discrepancy Constraint
    - QAC: Qualitative Alignment Constraint
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_score: Tensor of shape [B] or [B, 1]
            gt_score: Tensor of shape [B] or [B, 1]
        Returns:
            total_loss: Scalar loss value
        """
        if pred_score.dim() > 2:
            pred_score = pred_score.mean(1)
        if pred_score.dim() > 1:
            pred_score = pred_score.squeeze(-1)
        if gt_score.dim() > 1:
            gt_score = gt_score.squeeze(-1)

        # Expand to pairwise matrices
        delta_y = pred_score.unsqueeze(0) - pred_score.unsqueeze(1)  # [B, B]
        delta_t = gt_score.unsqueeze(0) - gt_score.unsqueeze(1)      # [B, B]

        # QDC: (delta_y - delta_t)^2
        qdc = torch.mean((delta_y - delta_t) ** 2)

        # QAC: - delta_y * sign(delta_t)
        qac = -torch.mean(delta_y * torch.sign(delta_t))

        total_loss = qdc + qac

        return total_loss
