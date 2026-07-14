"""Physically-grounded synthetic generator.

Produces hourly solar wind (Bt, Bz, By, V, n) with realistic quiet background
plus stochastic storm drivers (CME-like flux ropes and fast streams that turn
the IMF southward). Dst is produced by integrating the ring-current ODE with
process noise and mild nonlinear saturation, so the empirical model captures
most -- but not all -- of the signal, leaving genuine room for learning.

This lets the entire pipeline (features -> train -> evaluate -> serve) run with
zero internet, and is what the automated tests exercise. Swap in real data with
`--source omni` (training) and live NOAA serving on a networked machine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ..models import physics as phys


def _smooth_noise(rng, n, scale, corr=0.9):
    """AR(1) coloured noise for smooth background variability."""
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = corr * x[i - 1] + rng.normal(0, scale)
    return x


def simulate(hours: int = 24 * 365 * 6, seed: int = C.SEED) -> pd.DataFrame:
    """Simulate `hours` of hourly solar wind + Dst. Returns a tidy DataFrame."""
    rng = np.random.default_rng(seed)
    t = np.arange(hours)

    # --- Quiet background --------------------------------------------------
    # Slow solar-cycle-like modulation of activity level (drives storm rate).
    cycle = 0.5 * (1 + np.sin(2 * np.pi * t / (24 * 365 * 5.5)))  # 0..1
    v = 400 + _smooth_noise(rng, hours, 3.0) + 40 * cycle
    n = np.clip(5 + _smooth_noise(rng, hours, 0.15), 0.5, None)
    bt = np.clip(5 + _smooth_noise(rng, hours, 0.1), 0.5, None)
    bz = _smooth_noise(rng, hours, 0.3)
    by = _smooth_noise(rng, hours, 0.3)

    # --- Storm drivers -----------------------------------------------------
    # Poisson onsets, rate modulated by the activity cycle.
    base_rate = 0.0016  # per hour (~ one event every ~26 days at mid-cycle)
    onset_p = base_rate * (0.4 + 1.2 * cycle)
    onsets = rng.random(hours) < onset_p

    for i in np.where(onsets)[0]:
        dur = int(rng.integers(10, 40))          # storm main+recovery driver length
        if i + dur >= hours:
            continue
        strength = rng.uniform(0.4, 1.0)          # event severity
        idx = np.arange(dur)
        # Sheath: turbulent, then a smooth flux-rope rotation (sustained south Bz)
        sheath = int(dur * 0.3)
        env = np.ones(dur)
        # Field enhancement
        bt_boost = (10 + 25 * strength) * np.sin(np.pi * idx / dur) ** 0.5
        bt[i:i + dur] += bt_boost * env
        # Southward Bz: turbulent sheath + smooth rope minimum
        rope = -(bt_boost) * np.sin(np.pi * idx / dur)
        sheath_turb = rng.normal(0, 6 * strength, dur)
        sheath_turb[sheath:] *= 0.3
        bz[i:i + dur] += rope + sheath_turb
        by[i:i + dur] += rng.normal(0, 4 * strength, dur)
        # Fast stream: elevated speed and density pulse
        v[i:i + dur] += (150 + 300 * strength) * np.sin(np.pi * idx / dur) ** 0.7
        n[i:i + dur] += (5 + 15 * strength) * np.sin(np.pi * idx / dur) ** 2

    v = np.clip(v, 250, 1200)
    n = np.clip(n, 0.3, 90)
    bt = np.clip(bt, 0.3, 80)
    bz = np.clip(bz, -bt, bt)  # |Bz| <= Bt

    # --- Integrate ring current to get "true" Dst --------------------------
    vbs = phys.coupling_vbs(v, bz)
    pdyn = phys.dynamic_pressure(n, v)
    q = phys.obm_injection(vbs)
    tau = phys.obm_tau(vbs)

    dst_star = np.zeros(hours)
    proc = _smooth_noise(rng, hours, 0.8, corr=0.7)  # unmodelled dynamics
    for k in range(1, hours):
        # mild nonlinear saturation of injection at strong driving (nonlinear
        # departure from the linear empirical Q -> room for a learned model)
        q_eff = q[k - 1] * (1.0 - 0.15 * np.tanh(-q[k - 1] / 30.0))
        d = q_eff - dst_star[k - 1] / tau[k - 1] + proc[k]
        dst_star[k] = dst_star[k - 1] + d
    dst = phys.pressure_correction(dst_star, pdyn)
    dst += rng.normal(0, 1.5, hours)  # measurement noise on the index
    dst = np.clip(dst, -600, 60)

    # --- Observation noise on the solar wind the model actually sees --------
    bt_obs = np.clip(bt + rng.normal(0, 0.4, hours), 0.3, None)
    bz_obs = bz + rng.normal(0, 0.5, hours)
    bz_obs = np.clip(bz_obs, -bt_obs, bt_obs)  # keep |Bz| <= Bt physical
    obs = pd.DataFrame({
        "bt": bt_obs,
        "bz": bz_obs,
        "by": by + rng.normal(0, 0.5, hours),
        "v": v + rng.normal(0, 8, hours),
        "n": np.clip(n + rng.normal(0, 0.3, hours), 0.1, None),
        "dst": dst,
    })
    obs.index = pd.date_range("2015-01-01", periods=hours, freq="h", name="time")
    return obs


if __name__ == "__main__":  # quick sanity check
    df = simulate(24 * 90)
    print(df.describe().round(2))
    print("storm hours (Dst<-50):", int((df.dst < -50).sum()))
