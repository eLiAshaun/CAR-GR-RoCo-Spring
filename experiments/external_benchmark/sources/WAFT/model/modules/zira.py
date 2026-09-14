import torch.nn as nn


class ZIRA(nn.Module):
    """Zero-initialized residual adapter for the fused H/2 feature."""

    def __init__(self, channels: int = 64, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm = nn.GroupNorm(groups, channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        return x + self.conv2(self.act(self.norm(self.conv1(x))))
