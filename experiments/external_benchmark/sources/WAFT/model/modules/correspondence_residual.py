import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrespondenceResidualInjection(nn.Module):
    """Project normalized feature disagreement into the refinement input."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.diff_proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.diff_proj.weight)
        nn.init.zeros_(self.diff_proj.bias)

    def forward(self, refine_input, fmap1, warped_fmap2):
        diff = torch.abs(
            F.normalize(fmap1, dim=1, eps=1e-6)
            - F.normalize(warped_fmap2, dim=1, eps=1e-6)
        )
        return refine_input + self.diff_proj(diff)
