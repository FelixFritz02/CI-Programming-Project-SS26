import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQNNetwork(nn.Module):
    """
    Einfaches Feed-Forward-Netz zur Q-Wert-Approximation.

    Architektur: input → hidden_dims[0] → hidden_dims[1] → ... → output
    (Default hidden_dims=(32, 16), wie bisher fest verdrahtet.)
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple = (32, 16)):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU()]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)