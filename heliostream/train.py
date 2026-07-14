"""Training: heteroscedastic Gaussian NLL with optional storm up-weighting,
chronological early stopping, and a simple on-disk model registry."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import config as C
from .data import features as F
from .models.baseline import GRUBaseline
from .models.hybrid import PhysicsInformedDst

MODELS = {"hybrid": PhysicsInformedDst, "gru": GRUBaseline}


def gaussian_nll(mean, logvar, y, storm_weight=0.0):
    inv = torch.exp(-logvar)
    nll = 0.5 * (logvar + (y - mean) ** 2 * inv)
    if storm_weight > 0:
        w = 1.0 + storm_weight * torch.clamp(-(y - C.STORM_THRESHOLD) /
                                             abs(C.STORM_THRESHOLD), min=0)
        nll = nll * w
    return nll.mean()


def _tensors(df, norm):
    X, y, dst0, swf, idx = F.make_windows(df, norm)
    return (torch.from_numpy(X), torch.from_numpy(y),
            torch.from_numpy(dst0), torch.from_numpy(swf))


def train_model(df, model_name="hybrid", epochs=40, batch=256, lr=1e-3,
                storm_weight=1.0, patience=6, hidden=64, seed=C.SEED, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    df = F.engineer(df)
    tr, va, te = F.time_split(df)
    norm = F.fit_normalizer(tr)

    Xtr, ytr, d0tr, swtr = _tensors(tr, norm)
    Xva, yva, d0va, swva = _tensors(va, norm)

    dl = DataLoader(TensorDataset(Xtr, ytr, d0tr, swtr), batch_size=batch,
                    shuffle=True, drop_last=False)

    model = MODELS[model_name](hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)

    best_val, best_state, bad = float("inf"), None, 0
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        tl = 0.0
        for xb, yb, d0b, swb in dl:
            opt.zero_grad()
            mean, logvar = model(xb, d0b, swb)
            loss = gaussian_nll(mean, logvar, yb, storm_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl += loss.item() * len(xb)
        tl /= len(Xtr)

        model.eval()
        with torch.no_grad():
            mv, lv = model(Xva, d0va, swva)
            vloss = gaussian_nll(mv, lv, yva, storm_weight).item()
            vrmse = torch.sqrt(((mv - yva) ** 2).mean()).item()
        sched.step(vloss)
        hist.append({"epoch": ep, "train_nll": tl, "val_nll": vloss, "val_rmse": vrmse})
        if verbose:
            print(f"[{model_name}] ep{ep:02d} train_nll={tl:.3f} "
                  f"val_nll={vloss:.3f} val_rmse={vrmse:.2f}nT")
        if vloss < best_val - 1e-4:
            best_val, best_state, bad = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop @ ep{ep}")
                break

    model.load_state_dict(best_state)
    return model, norm, hist


def save(model, norm, model_name, hist, extra=None):
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = C.MODEL_DIR / f"{model_name}.pt"
    torch.save(model.state_dict(), path)
    meta = {
        "model": model_name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "features": C.FEATURE_COLS,
        "lookback": C.LOOKBACK,
        "horizons": C.HORIZONS,
        "normalizer": norm.to_dict(),
        "history": hist,
    }
    if extra:
        meta.update(extra)
    (C.MODEL_DIR / f"{model_name}.json").write_text(json.dumps(meta, indent=2))
    return path


def load(model_name="hybrid", hidden=64):
    meta = json.loads((C.MODEL_DIR / f"{model_name}.json").read_text())
    model = MODELS[model_name](hidden=hidden)
    model.load_state_dict(torch.load(C.MODEL_DIR / f"{model_name}.pt", weights_only=True))
    model.eval()
    norm = F.Normalizer.from_dict(meta["normalizer"])
    return model, norm, meta
