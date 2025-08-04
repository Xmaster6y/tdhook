"""
Models for benchmarking.
"""

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, height: int = 12, width: int = 10):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(height)])
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x
