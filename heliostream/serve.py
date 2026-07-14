"""Live serving: build a nowcast from the most recent solar-wind window and
expose it via a small FastAPI app that also serves the dashboard.

Run:  heliostream serve            (live NOAA feed)
      heliostream serve --demo     (offline synthetic feed)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import config as C
from .data import features as F
from . import train as T

ALERTS = [
    (-30,  {"level": 0, "label": "Quiet", "color": "#3ee6c4"}),
    (-50,  {"level": 1, "label": "Unsettled", "color": "#8ab4ff"}),
    (-100, {"level": 2, "label": "Minor storm (G1)", "color": "#c9a227"}),
    (-200, {"level": 3, "label": "Moderate\u2013strong storm (G2\u2013G3)", "color": "#ff8c42"}),
    (-9999, {"level": 4, "label": "Severe storm (G4\u2013G5)", "color": "#ff4d6d"}),
]


def alert_for(min_dst: float):
    for thresh, meta in ALERTS:
        if min_dst >= thresh:
            return meta
    return ALERTS[-1][1]


def _factors():
    p = C.MODEL_DIR / "calibration.json"
    if p.exists():
        return np.array(json.loads(p.read_text())["factors_per_h"])
    from scipy.stats import norm as _n
    return np.full(len(C.HORIZONS), float(_n.ppf(0.95)))


def build_nowcast(model, norm, window_df, dst0, source="noaa-live", factors=None):
    """window_df: raw hourly RAW_COLS, length == LOOKBACK. Returns nowcast dict."""
    if factors is None:
        factors = _factors()
    win = F.engineer(window_df)
    x = norm.transform(win)[None, -C.LOOKBACK:, :].astype("float32")
    last = win.iloc[-1]
    swf = np.array([[last["v"], last["n"], last["bz"]]], dtype="float32")
    d0 = np.array([dst0], dtype="float32")

    model.eval()
    with torch.no_grad():
        mean, logvar = model(torch.from_numpy(x), torch.from_numpy(d0),
                             torch.from_numpy(swf))
    mean = mean.numpy()[0]
    std = np.exp(0.5 * logvar.numpy()[0])

    forecast = []
    for j, h in enumerate(C.HORIZONS):
        lo = float(mean[j] - factors[j] * std[j])
        hi = float(mean[j] + factors[j] * std[j])
        forecast.append({"h": h, "mean": float(mean[j]), "lo": lo, "hi": hi})

    min_dst = float(np.min([f["mean"] for f in forecast]))
    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "dst0": float(dst0),
        "current_solar_wind": {
            "bz": float(last["bz"]), "bt": float(last["bt"]),
            "v": float(last["v"]), "n": float(last["n"]),
            "vbs": float(last["vbs"]), "pdyn": float(last["pdyn"]),
        },
        "horizons": C.HORIZONS,
        "forecast": forecast,
        "min_dst": min_dst,
        "alert": alert_for(min_dst),
        "model": getattr(model, "_name", "hybrid"),
        "conformal_target": 0.9,
    }


# --- demo feed ------------------------------------------------------------
class _DemoFeed:
    """Advances a pointer through a synthetic series so the dashboard 'moves'."""
    def __init__(self):
        from .data import synthetic
        self.df = synthetic.simulate(24 * 400, seed=7)
        self.ptr = C.LOOKBACK + 100

    def window(self):
        self.ptr += 1
        if self.ptr >= len(self.df) - 1:
            self.ptr = C.LOOKBACK + 100
        w = self.df.iloc[self.ptr - C.LOOKBACK: self.ptr]
        dst0 = float(self.df["dst"].iloc[self.ptr - 1])
        return w[C.RAW_COLS].copy(), dst0


def make_app(model_name="hybrid", demo=False):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    model, norm, meta = T.load(model_name)
    model._name = model_name
    factors = _factors()
    dash = (Path(__file__).resolve().parent.parent / "dashboard" / "index.html").read_text(encoding="utf-8")
    feed = _DemoFeed() if demo else None

    app = FastAPI(title="Heliostream", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return dash

    @app.get("/api/nowcast")
    def nowcast():
        try:
            if demo:
                window, dst0 = feed.window()
                src = "demo-synthetic"
            else:
                from .data import noaa_live
                window, dst0 = noaa_live.fetch_window()
                src = "noaa-live"
            return JSONResponse(build_nowcast(model, norm, window, dst0,
                                              source=src, factors=factors))
        except Exception as e:  # surface errors to the dashboard cleanly
            return JSONResponse({"error": str(e)}, status_code=503)

    return app


def run(model_name="hybrid", demo=False, host="127.0.0.1", port=8000):
    import uvicorn
    uvicorn.run(make_app(model_name, demo), host=host, port=port)
