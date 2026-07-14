"""Evaluation, calibration, and baselines.

Reports overall and storm-time RMSE per horizon, interval calibration, and
applies split-conformal calibration for coverage guarantees. Includes
persistence and empirical O'Brien-McPherron baselines for context.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from scipy.stats import norm as normal  # only used for the z multiplier

from . import config as C
from .data import features as F
from .models import physics as phys


# --------------------------------------------------------------------------
def predict(model, df, norm):
    """Return mean (N,H), std (N,H), y (N,H) for a fitted torch model."""
    df = F.engineer(df)
    X, y, dst0, swf, idx = F.make_windows(df, norm)
    model.eval()
    with torch.no_grad():
        mean, logvar = model(torch.from_numpy(X), torch.from_numpy(dst0),
                             torch.from_numpy(swf))
    std = np.exp(0.5 * logvar.numpy())
    return mean.numpy(), std, y, idx


def obm_forecast(df, norm=None):
    """Empirical physics baseline: persist the origin driver and integrate OBM."""
    df = F.engineer(df)
    X, y, dst0, swf, idx = F.make_windows(df, norm or F.fit_normalizer(df))
    preds = np.zeros_like(y)
    for i in range(len(dst0)):
        v0, n0, bz0 = swf[i]
        seq_v = np.full(C.MAX_HORIZON + 1, v0)
        seq_n = np.full(C.MAX_HORIZON + 1, n0)
        seq_bz = np.full(C.MAX_HORIZON + 1, bz0)
        traj = phys.integrate_obm(seq_v, seq_n, seq_bz, float(dst0[i]))
        preds[i] = [traj[h] for h in C.HORIZONS]
    return preds, y


def persistence_forecast(df, norm=None):
    df = F.engineer(df)
    X, y, dst0, swf, idx = F.make_windows(df, norm or F.fit_normalizer(df))
    return np.repeat(dst0[:, None], len(C.HORIZONS), axis=1), y


# --------------------------------------------------------------------------
def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def metrics(mean, y, std=None, tag=""):
    out = {"tag": tag}
    out["rmse_all"] = _rmse(mean, y)
    out["mae_all"] = float(np.mean(np.abs(mean - y)))
    storm = y < C.STORM_THRESHOLD
    out["rmse_storm"] = _rmse(mean[storm], y[storm]) if storm.any() else None
    out["n_storm"] = int(storm.sum())
    out["rmse_per_h"] = [_rmse(mean[:, j], y[:, j]) for j in range(y.shape[1])]
    if std is not None:
        # 90% nominal coverage of a Gaussian predictive interval
        z = normal.ppf(0.95)
        lo, hi = mean - z * std, mean + z * std
        out["coverage90"] = float(np.mean((y >= lo) & (y <= hi)))
    return out


# --------------------------------------------------------------------------
def conformal_factors(mean_cal, std_cal, y_cal, alpha=0.1):
    """Split-conformal: per-horizon multiplier q_h on std so that
    mean +/- q_h*std attains >= 1-alpha marginal coverage on future data."""
    H = y_cal.shape[1]
    factors = np.zeros(H)
    for j in range(H):
        s = np.abs(y_cal[:, j] - mean_cal[:, j]) / np.maximum(std_cal[:, j], 1e-6)
        n = len(s)
        k = int(np.ceil((n + 1) * (1 - alpha))) - 1
        k = min(max(k, 0), n - 1)
        factors[j] = np.sort(s)[k]
    return factors


def coverage(mean, std, y, factors):
    lo = mean - factors * std
    hi = mean + factors * std
    return float(np.mean((y >= lo) & (y <= hi)))


# --------------------------------------------------------------------------
def full_report(model, norm, df_val, df_test, alpha=0.1):
    """Evaluate model on test, calibrate intervals on val, compare to baselines."""
    mean_t, std_t, y_t, _ = predict(model, df_test, norm)
    mean_v, std_v, y_v, _ = predict(model, df_val, norm)

    factors = conformal_factors(mean_v, std_v, y_v, alpha=alpha)
    cov_before = metrics(mean_t, y_t, std_t)["coverage90"]
    cov_after = coverage(mean_t, std_t, y_t, factors)

    rep = {
        "model": metrics(mean_t, y_t, std_t, tag="model"),
        "conformal": {
            "alpha": alpha,
            "target_coverage": 1 - alpha,
            "coverage_gaussian": cov_before,
            "coverage_conformal": cov_after,
            "factors_per_h": factors.tolist(),
        },
    }
    # Baselines
    p_pers, _ = persistence_forecast(df_test, norm)
    p_obm, _ = obm_forecast(df_test, norm)
    rep["persistence"] = metrics(p_pers, y_t, tag="persistence")
    rep["obm_empirical"] = metrics(p_obm, y_t, tag="obm_empirical")
    return rep, (mean_t, std_t, y_t, factors)


def save_calibration(factors, alpha=0.1):
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (C.MODEL_DIR / "calibration.json").write_text(
        json.dumps({"alpha": alpha, "factors_per_h": list(map(float, factors))},
                   indent=2))


def save_plots(model, norm, df_test, factors, prefix="hybrid"):
    """Storm-trace (h=1) with conformal band + a coverage reliability curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean, std, y, idx = predict(model, df_test, norm)
    dfe = F.engineer(df_test)

    # --- worst storm stretch (by true h=1 minimum) ---
    j = 0  # horizon +1h
    center = int(np.argmin(y[:, j]))
    a, b = max(0, center - 150), min(len(y), center + 150)
    z = normal.ppf(0.95)
    lo = mean[a:b, j] - factors[j] * std[a:b, j]
    hi = mean[a:b, j] + factors[j] * std[a:b, j]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(range(b - a), lo, hi, color="#7c5cff", alpha=0.25,
                    label="90% conformal interval")
    ax.plot(mean[a:b, j], color="#7c5cff", lw=1.6, label="forecast +1h")
    ax.plot(y[a:b, j], color="#e6e6e6", lw=1.2, label="observed Dst")
    ax.axhline(C.STORM_THRESHOLD, color="#ff8c42", ls="--", lw=0.8)
    ax.set_title("Worst storm in test window (+1h nowcast)")
    ax.set_xlabel("hours"); ax.set_ylabel("Dst (nT)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    p1 = C.REPORT_DIR / f"{prefix}_storm_trace.png"
    fig.savefig(p1, dpi=130); plt.close(fig)

    # --- reliability curve (nominal vs empirical coverage) ---
    levels = np.linspace(0.5, 0.99, 12)
    emp = []
    for lv in levels:
        zc = normal.ppf(0.5 + lv / 2)
        f = np.array([conformal_factors(mean, std, y, alpha=1 - lv)])  # per-h
        cov = coverage(mean, std, y, f[0])
        emp.append(cov)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0.5, 1], [0.5, 1], color="#555", ls="--", lw=1)
    ax.plot(levels, emp, "o-", color="#7c5cff")
    ax.set_title("Coverage reliability (conformal)")
    ax.set_xlabel("nominal"); ax.set_ylabel("empirical")
    fig.tight_layout()
    p2 = C.REPORT_DIR / f"{prefix}_reliability.png"
    fig.savefig(p2, dpi=130); plt.close(fig)
    return str(p1), str(p2)
