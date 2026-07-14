"""Live upstream solar wind from NOAA SWPC.

Pulls the rolling magnetometer and plasma JSON feeds (which automatically carry
whichever spacecraft SWPC has marked 'active' -- DSCOVR/ACE today, SOLAR-1/IMAP
as the 2026 migration completes), resamples to hourly, and returns the most
recent LOOKBACK-hour window ready for the model. Also makes a best-effort fetch
of a real-time Dst anchor for the physics initial condition.

Requires internet; the build sandbox cannot reach NOAA, so this path is
exercised on a networked machine (or via `serve --demo`, which uses the
synthetic simulator instead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C

BASE = "https://services.swpc.noaa.gov/products/solar-wind"
MAG = f"{BASE}/mag-1-day.json"
PLASMA = f"{BASE}/plasma-1-day.json"
# Candidate real-time Dst-like products (tried in order; optional).
DST_CANDIDATES = [
    "https://services.swpc.noaa.gov/products/kyoto-dst.json",
]


def _table(url, timeout=30):
    import requests
    rows = requests.get(url, timeout=timeout).json()
    header, data = rows[0], rows[1:]
    return pd.DataFrame(data, columns=header)


def _to_hourly(df, time_col, num_cols):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.tz_localize(None)
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.set_index(time_col).sort_index()
    return df[num_cols].resample("h").mean()


def fetch_recent_dst():
    """Best-effort real-time Dst [nT]; None if unavailable."""
    import requests
    for url in DST_CANDIDATES:
        try:
            tbl = _table(url)
            val_col = [c for c in tbl.columns if c.lower() not in
                       ("time_tag", "time")][-1]
            return float(pd.to_numeric(tbl[val_col], errors="coerce").dropna().iloc[-1])
        except Exception:
            continue
    return None


def fetch_window(lookback=C.LOOKBACK):
    """Return (window_df, dst0). window_df has RAW_COLS at hourly cadence,
    length >= lookback (most recent hours). dst0 is the physics init state."""
    mag = _table(MAG)          # time_tag, bx_gsm, by_gsm, bz_gsm, lon_gsm, lat_gsm, bt
    pla = _table(PLASMA)       # time_tag, density, speed, temperature

    mag_h = _to_hourly(mag, "time_tag", ["by_gsm", "bz_gsm", "bt"])
    pla_h = _to_hourly(pla, "time_tag", ["density", "speed"])
    df = mag_h.join(pla_h, how="inner").rename(columns={
        "bz_gsm": "bz", "by_gsm": "by", "density": "n", "speed": "v"})
    df = df[["bt", "bz", "by", "v", "n"]].interpolate(limit=3).dropna()

    if len(df) < lookback:
        raise RuntimeError(f"Only {len(df)} clean hours from NOAA; "
                           f"need {lookback}. Try again shortly.")
    window = df.iloc[-lookback:].copy()

    dst0 = fetch_recent_dst()
    if dst0 is None:
        # Fall back to a quiet-time anchor; the forecast is driven mainly by the
        # solar-wind window, dst0 is only the ODE initial condition.
        dst0 = -15.0
    return window, float(dst0)
