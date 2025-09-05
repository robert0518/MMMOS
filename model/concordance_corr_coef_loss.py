import torch
import torch.nn as nn

class ConcordanceCorrCoefLoss(nn.Module):
    """
    Concordance Correlation Coefficient (CCC) Loss.

    CCC = (2 * cov(x, y)) / (var(x) + var(y) + (mean_x - mean_y)^2)
    Loss = 1 - CCC
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred_score: torch.Tensor, gt_score) -> torch.Tensor:
        if pred_score.dim() > 1:
            pred_score = pred_score.squeeze(-1)
        if gt_score.dim() > 1:
            gt_score = gt_score.squeeze(-1)

        # means
        mean_pred = torch.mean(pred_score)
        mean_gt = torch.mean(gt_score)

        # variances
        var_pred = torch.var(pred_score, unbiased=True)
        var_gt = torch.var(gt_score, unbiased=True)

        # covariance
        cov = torch.mean((pred_score - mean_pred) * (gt_score - mean_gt))

        # CCC calculation
        ccc = (2 * cov) / (
            var_pred + var_gt + (mean_pred - mean_gt) ** 2 + self.eps
        )

        # CCC loss: 1 - CCC
        return 1.0 - ccc
