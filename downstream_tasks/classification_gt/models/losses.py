import numpy as np
import torch
import torch.nn as nn

class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super().__init__()
        if isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights)

        self.weights = weights/torch.sum(weights)
        self.bce = nn.BCEWithLogitsLoss(reduction="none" if weights is not None else "mean")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        loss = self.bce(input=logits, target=targets.to(torch.float32))

        if self.weights is None:
            return loss
        else:
            weighted_loss = loss * self.weights.to(loss.device)
            return weighted_loss.mean()
