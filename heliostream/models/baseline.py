"""Pure-GRU heteroscedastic baseline (no physics structure)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .. import config as C


class GRUBaseline(nn.Module):
    def __init__(self, n_features=len(C.FEATURE_COLS), hidden=64, layers=2,
                 horizons=C.HORIZONS, dropout=0.1):
        super().__init__()
        self.horizons = list(horizons)
        H = len(self.horizons)
        self.gru = nn.GRU(n_features, hidden, num_layers=layers,
                          batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.mean_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, H))
        self.logvar_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, H))

    def forward(self, x, dst0=None, swf=None):
        _, h = self.gru(x)
        z = h[-1]
        mean = self.mean_head(z)
        logvar = self.logvar_head(z).clamp(-6, 8)
        return mean, logvar
