"""Shared trajectory horizons and the imitation-policy architecture."""
import numpy as np

HORIZONS = np.array([1., 2., 3., 4., 5.], np.float64)
GOAL_HORIZON = 8.0


def make_model(k=5):
    """Build the five-channel BEV trajectory distribution network."""
    import torch
    from torch import nn

    class TrajectoryPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(5, 24, 5, 2, 2), nn.ReLU(),
                nn.Conv2d(24, 48, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(48, 64, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(64, 96, 3, 2, 1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Sequential(
                nn.Linear(96 + 3 + 2, 192),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(192, 4 * k),
            )

        def forward(self, bev, state, goal):
            features = self.encoder(bev).flatten(1)
            output = self.head(torch.cat([features, state, goal], dim=1))
            mean = output[:, :2 * k].reshape(-1, k, 2)
            log_std = output[:, 2 * k:].reshape(-1, k, 2).clamp(-4, 2)
            return mean, log_std

    return TrajectoryPolicy()
