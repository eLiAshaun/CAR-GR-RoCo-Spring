import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleResidualInjection(nn.Module):
    """Reuse the already-computed H/4 and H/8 ResNet features."""

    def __init__(self, target_channels: int = 64, use_s4: bool = False):
        super().__init__()
        self.proj_s2 = nn.Conv2d(128, target_channels, 1)
        self.proj_s3 = nn.Conv2d(256, target_channels, 1)
        self.beta_s2 = nn.Parameter(torch.zeros(()))
        self.beta_s3 = nn.Parameter(torch.zeros(()))
        self.use_s4 = use_s4
        if use_s4:
            self.proj_s4 = nn.Conv2d(512, target_channels, 1)
            self.beta_s4 = nn.Parameter(torch.zeros(()))

    def forward(self, base, features):
        _, f2, f3, f4 = features
        size = base.shape[-2:]
        result = base
        result = result + self.beta_s2 * F.interpolate(
            self.proj_s2(f2), size=size, mode="bilinear", align_corners=False
        )
        result = result + self.beta_s3 * F.interpolate(
            self.proj_s3(f3), size=size, mode="bilinear", align_corners=False
        )
        if self.use_s4:
            result = result + self.beta_s4 * F.interpolate(
                self.proj_s4(f4), size=size, mode="bilinear", align_corners=False
            )
        return result
