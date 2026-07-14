"""Physics-informed hybrid forecaster.

A GRU encodes the recent solar-wind window into a context vector, from which
small heads emit the ring-current *injection* Q and *decay* tau. The Dst state
is then rolled forward by integrating the O'Brien-McPherron ODE

    dDst*/dt = Q - Dst*/tau

so the forecast trajectory is physically shaped (injection then exponential
recovery) by construction. A bounded neural residual lets the network correct
structured departures from the empirical law without letting it ignore the
physics. A separate head emits per-horizon predictive variance (heteroscedastic
Gaussian NLL), which downstream conformal calibration then makes coverage-exact.

Why this matters for the project's research angle: the inductive bias improves
extreme-storm behaviour and degrades gracefully under noisy / missing drivers,
which pure black-box sequence models handle poorly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config as C


def _remove_pressure(dst, pdyn):
    return dst - C.PRESS_B * torch.sqrt(pdyn.clamp(min=0)) + C.PRESS_C


def _apply_pressure(dst_star, pdyn):
    return dst_star + C.PRESS_B * torch.sqrt(pdyn.clamp(min=0)) - C.PRESS_C


class PhysicsInformedDst(nn.Module):
    def __init__(self, n_features=len(C.FEATURE_COLS), hidden=64, layers=2,
                 horizons=C.HORIZONS, dropout=0.1, residual_cap=25.0):
        super().__init__()
        self.horizons = list(horizons)
        self.max_h = max(self.horizons)
        self.residual_cap = residual_cap
        self.gru = nn.GRU(n_features, hidden, num_layers=layers,
                          batch_first=True, dropout=dropout if layers > 1 else 0.0)
        # +2 driver features (vbs0, pdyn0) appended to the context
        ctx = hidden + 2
        self.q_head = nn.Sequential(nn.Linear(ctx, hidden), nn.GELU(),
                                    nn.Linear(hidden, 1))
        self.tau_head = nn.Sequential(nn.Linear(ctx, hidden), nn.GELU(),
                                      nn.Linear(hidden, 1))
        self.res_head = nn.Sequential(nn.Linear(ctx, hidden), nn.GELU(),
                                      nn.Linear(hidden, len(self.horizons)))
        self.logvar_head = nn.Sequential(nn.Linear(ctx, hidden), nn.GELU(),
                                         nn.Linear(hidden, len(self.horizons)))

    def forward(self, x, dst0, swf):
        """x:(B,L,F)  dst0:(B,)  swf:(B,3)=[v,n,bz] at origin (raw units)."""
        _, h = self.gru(x)
        z = h[-1]                                  # (B, hidden)
        v0, n0, bz0 = swf[:, 0], swf[:, 1], swf[:, 2]
        bs0 = torch.clamp(-bz0, min=0)
        vbs0 = v0 * bs0 * 1e-3                      # mV/m
        pdyn0 = C.PROTON_MASS_FACTOR * n0 * v0 ** 2
        z = torch.cat([z, vbs0.unsqueeze(1), pdyn0.unsqueeze(1)], dim=1)

        # Physics params (constrained): Q <= 0 (injection), tau > floor
        Q = -F.softplus(self.q_head(z)).squeeze(1)          # (B,) nT/hr
        tau = F.softplus(self.tau_head(z)).squeeze(1) + 0.7  # (B,) hr
        residual = self.residual_cap * torch.tanh(self.res_head(z))  # (B,H)
        logvar = self.logvar_head(z).clamp(-6, 8)           # (B,H)

        # Integrate ODE forward (RK2, dt=1h), persisting the origin driver.
        dst_star = _remove_pressure(dst0, pdyn0)            # (B,)
        means = []
        hi = 0
        for step in range(1, self.max_h + 1):
            k1 = Q - dst_star / tau
            mid = dst_star + 0.5 * k1
            k2 = Q - mid / tau
            dst_star = dst_star + k2
            if step in self.horizons:
                obs = _apply_pressure(dst_star, pdyn0) + residual[:, hi]
                means.append(obs)
                hi += 1
        mean = torch.stack(means, dim=1)                    # (B,H)
        return mean, logvar

    @torch.no_grad()
    def physics_params(self, x, dst0, swf):
        """Expose learned Q, tau for interpretability panels."""
        _, h = self.gru(x)
        z = h[-1]
        v0, n0, bz0 = swf[:, 0], swf[:, 1], swf[:, 2]
        vbs0 = v0 * torch.clamp(-bz0, min=0) * 1e-3
        pdyn0 = C.PROTON_MASS_FACTOR * n0 * v0 ** 2
        z = torch.cat([z, vbs0.unsqueeze(1), pdyn0.unsqueeze(1)], dim=1)
        Q = -F.softplus(self.q_head(z)).squeeze(1)
        tau = F.softplus(self.tau_head(z)).squeeze(1) + 0.7
        return Q, tau
