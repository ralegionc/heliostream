"""Record sources feeding the bus. Each yields canonical record dicts:
{time, bt, bz, by, v, n, dst, source}.

Reuses the heliostream package so the simulator, NOAA live feed, and OMNI
backfill are shared with the model side (single source of truth).
"""
from __future__ import annotations

import pandas as pd


def _frame_to_records(df: pd.DataFrame, source: str):
    df = df.reset_index().rename(columns={df.index.name or "index": "time"})
    for _, r in df.iterrows():
        yield {
            "time": pd.Timestamp(r["time"]).isoformat(),
            "bt": _f(r.get("bt")), "bz": _f(r.get("bz")), "by": _f(r.get("by")),
            "v": _f(r.get("v")), "n": _f(r.get("n")), "dst": _f(r.get("dst")),
            "source": source,
        }


def _f(x):
    return None if x is None or pd.isna(x) else float(x)


def synthetic_records(hours=24 * 30, seed=0):
    """Streaming stand-in: replay a synthetic solar-wind series as records."""
    from heliostream.data import synthetic
    df = synthetic.simulate(hours=hours, seed=seed)
    yield from _frame_to_records(df, "synthetic")


def noaa_records():
    """Most recent real solar-wind hours from NOAA (Dst is left null; the final
    Dst index is backfilled separately by OMNI, mirroring real operations)."""
    from heliostream.data import noaa_live
    window, dst0 = noaa_live.fetch_window()
    window = window.copy()
    window["dst"] = None
    window.iloc[-1, window.columns.get_loc("dst")] = dst0  # best-effort current
    yield from _frame_to_records(window, "noaa")


def omni_backfill_records(start="2016-01-01", stop="2024-01-01"):
    """Historical batch backfill (solar wind + definitive Dst) from NASA OMNI."""
    from heliostream.data import omni
    df = omni.fetch(start=start, stop=stop)
    yield from _frame_to_records(df, "omni")
