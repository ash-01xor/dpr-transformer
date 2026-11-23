import torch
from torch import nn


class DPRLoss(nn.Module):
    def forward(self, queries, passages, positives):
        """Negative log probability assigned to the relevant passages in each row."""
        scores = (queries @ passages.T).float()
        positive_scores = scores.masked_fill(~positives, float("-inf"))
        return (torch.logsumexp(scores, dim=1) - torch.logsumexp(positive_scores, dim=1)).mean()
