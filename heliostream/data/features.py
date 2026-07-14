"""Feature engineering and supervised windowing.

Turns a tidy solar-wind + Dst frame into normalized (lookback window -> multi-
horizon Dst) samples. All normalization statistics are fit on the training span
only and reused downstream to prevent leakage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config as C
from ..models import physics as phys


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived physical features (coupling, pressure, clock angle)."""
    out = df.copy()
    out["bs"] = np.clip(-out["bz"], 0.0, None)
    out["vbs"] = phys.coupling_vbs(out["v"].values, out["bz"].values)
    out["pdyn"] = phys.dynamic_pressure(out["n"].values, out["v"].values)
    clock = np.arctan2(out["by"].values, out["bz"].values)  # IMF clock angle
    out["sin_clock_half4"] = np.sin(clock / 2.0) ** 4       # Newell-style coupling shape
    return out


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray
    cols: list

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return (df[self.cols].values - self.mean) / self.std

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "cols": self.cols}

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d["mean"]), np.array(d["std"]), d["cols"])


def fit_normalizer(df: pd.DataFrame, cols=C.FEATURE_COLS) -> Normalizer:
    mean = df[cols].mean().values
    std = df[cols].std().values + 1e-6
    return Normalizer(mean, std, list(cols))


def make_windows(
    df: pd.DataFrame,
    norm: Normalizer,
    lookback: int = C.LOOKBACK,
    horizons=C.HORIZONS,
):
    """Build arrays for supervised learning.

    Returns
    -------
    X    : (N, lookback, F)  normalized feature windows
    y    : (N, H)            future Dst targets [nT] at each horizon
    dst0 : (N,)              Dst at forecast origin t0 [nT] (physics init state)
    swf  : (N, 3)            [v, n, bz] at t0, raw units (physics rollout driver)
    idx  : (N,)              integer position of t0 in df (for backtest/plots)
    """
    feats = norm.transform(df)                 # (T, F)
    dst = df[C.TARGET_COL].values
    raw = df[["v", "n", "bz"]].values
    T = len(df)
    Hmax = max(horizons)
    starts = np.arange(lookback - 1, T - Hmax)
    X, y, dst0, swf, idx = [], [], [], [], []
    for s in starts:
        X.append(feats[s - lookback + 1: s + 1])
        y.append([dst[s + h] for h in horizons])
        dst0.append(dst[s])
        swf.append(raw[s])
        idx.append(s)
    return (
        np.asarray(X, np.float32),
        np.asarray(y, np.float32),
        np.asarray(dst0, np.float32),
        np.asarray(swf, np.float32),
        np.asarray(idx, np.int64),
    )


def time_split(df: pd.DataFrame, train=0.7, val=0.15):
    """Chronological split (no shuffling) -> (train, val, test) frames."""
    n = len(df)
    a, b = int(n * train), int(n * (train + val))
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()
