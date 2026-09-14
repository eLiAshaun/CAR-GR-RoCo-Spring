import torch
import torch.nn as nn


class UncertaintyDampedGate(nn.Module):
    def __init__(self, hidden_dim: int = 64, info_dim: int = 4, amplitude: float = 0.25):
        super().__init__()
        self.amplitude = amplitude
        self.step_gate = nn.Conv2d(hidden_dim + info_dim, 2, 1)
        nn.init.zeros_(self.step_gate.weight)
        nn.init.zeros_(self.step_gate.bias)

    def forward(self, flow, delta, hidden, info):
        gate = 1.0 + self.amplitude * torch.tanh(
            self.step_gate(torch.cat([hidden, info], dim=1))
        )
        return flow + gate * delta
