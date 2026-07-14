"""Expanding-window walk-forward backtest.

Trains on all data up to a cut point and tests on the next block, stepping
forward. This is the leakage-safe way to estimate operational skill and to
expose degradation under distribution shift (e.g. across the solar cycle).
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .data import features as F
from . import train as T
from . import evaluate as E


def walk_forward(df, model_name="hybrid", folds=4, epochs=25,
                 min_train_frac=0.4, storm_weight=1.0, verbose=True):
    df = F.engineer(df)
    n = len(df)
    start = int(n * min_train_frac)
    block = (n - start) // folds
    results = []
    for k in range(folds):
        cut = start + k * block
        test = df.iloc[cut: cut + block]
        train_pool = df.iloc[:cut]
        if len(test) < C.LOOKBACK + C.MAX_HORIZON + 10:
            break
        model, norm, _ = T.train_model(train_pool, model_name, epochs=epochs,
                                       storm_weight=storm_weight, verbose=False)
        mean, std, y, _ = E.predict(model, test, norm)
        m = E.metrics(mean, y, std, tag=f"fold{k}")
        m["train_hours"] = len(train_pool)
        m["test_start"] = str(df.index[cut])
        results.append(m)
        if verbose:
            print(f"fold {k}: train={len(train_pool)}h  "
                  f"rmse_all={m['rmse_all']:.2f}  "
                  f"rmse_storm={m['rmse_storm'] if m['rmse_storm'] else float('nan'):.2f}  "
                  f"cov90={m['coverage90']:.3f}")
    return results
