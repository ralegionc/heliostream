"""Historical training data from NASA/GSFC OMNI (hourly merged solar wind + Dst).

Primary path uses the official `hapiclient` against NASA CDAWeb's HAPI server,
which returns a typed array plus metadata (units, fill values) and handles the
wire format for us. Falls back to a raw CSV request if hapiclient is absent.

Dataset: OMNI2_H0_MRG1HR (hourly, 1963-present, definitive, time-shifted to the
bow-shock nose). Dst is the parameter `DST1800`. Requires internet; the sandbox
that generated this project cannot reach NASA, so validate on a networked
machine (e.g. the companion Colab notebook).
Docs: https://cdaweb.gsfc.nasa.gov/hapi/
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SERVER = "https://cdaweb.gsfc.nasa.gov/hapi"
DATASET = "OMNI2_H0_MRG1HR"

# our column -> candidate exact HAPI names (OMNI2 hourly suffixes many with 1800),
# then keyword fallbacks matched against name/description
WANT = {
    "bt":  ["F1800", "ABS_B1800", "F", "ABS_B"],
    "bz":  ["BZ_GSM1800", "BZ_GSM", "bz_gsm"],
    "by":  ["BY_GSM1800", "BY_GSM", "by_gsm"],
    "v":   ["V1800", "V", "flow_speed"],
    "n":   ["N1800", "N", "proton_density"],
    "dst": ["DST1800", "DST"],
}
# description keywords used only if name-matching fails
DESC = {
    "bt": ["field magnitude average"],
    "bz": ["bz", "gsm"],
    "by": ["by", "gsm"],
    "v":  ["flow speed"],
    "n":  ["ion number density", "proton density"],
    "dst": ["dst"],
}


def _resolve(param_names, descriptions=None):
    descriptions = descriptions or {}
    lower = {p.lower(): p for p in param_names}
    chosen = {}
    for col, cands in WANT.items():
        pick = next((lower[c.lower()] for c in cands if c.lower() in lower), None)
        if pick is None:  # loose contains-match on names
            pick = next((orig for lid, orig in lower.items()
                         if any(c.lower() in lid for c in cands)
                         and "sigma" not in lid and "rms" not in lid), None)
        if pick is None:  # last resort: description keywords
            kws = DESC.get(col, [])
            pick = next((n for n in param_names
                         if all(k in (descriptions.get(n, "") or "").lower() for k in kws)
                         and "sigma" not in n.lower()), None)
        if pick is None:
            raise KeyError(f"OMNI: no column for '{col}' among {param_names}")
        chosen[col] = pick
    return chosen


def _fetch_hapiclient(start, stop):
    from hapiclient import hapi
    meta = hapi(SERVER, DATASET)                       # metadata (all params)
    names = [p["name"] for p in meta["parameters"] if p["name"].lower() != "time"]
    descs = {p["name"]: p.get("description", "") for p in meta["parameters"]}
    chosen = _resolve(names, descs)
    fills = {p["name"]: p.get("fill") for p in meta["parameters"]}
    # HAPI requires requested params in metadata order; we read back by name.
    order = {p["name"]: i for i, p in enumerate(meta["parameters"])}
    plist = ",".join(sorted(set(chosen.values()), key=lambda nm: order[nm]))
    # options must be keyword args, not a positional dict (else hapi returns None)
    data, _ = hapi(SERVER, DATASET, plist, f"{start}T00:00:00Z",
                   f"{stop}T00:00:00Z", logging=False, usecache=False)
    tvals = data["Time"]
    if tvals.dtype.kind == "S":
        tvals = np.char.decode(tvals)
    elif tvals.dtype.kind == "O":
        tvals = tvals.astype(str)
    idx = pd.to_datetime(tvals, utc=True).tz_convert(None)
    out = pd.DataFrame(index=pd.Index(idx, name="time"))
    for col in WANT:
        name = chosen[col]
        vals = np.asarray(data[name], dtype="float64").ravel()
        f = fills.get(name)
        if f not in (None, "null"):
            try:
                vals = np.where(np.isclose(vals, float(f), rtol=1e-3), np.nan, vals)
            except (TypeError, ValueError):
                pass
        vals[vals > 9e4] = np.nan          # OMNI conventional fills (999.9, 9999.)
        out[col] = vals
    return out


def _fetch_csv(start, stop):
    import io, requests
    info = requests.get(f"{SERVER}/info", params={"dataset": DATASET}, timeout=60).json()
    names = [p["name"] for p in info["parameters"] if p["name"].lower() != "time"]
    descs = {p["name"]: p.get("description", "") for p in info["parameters"]}
    chosen = _resolve(names, descs)
    fills = {p["name"]: p.get("fill") for p in info["parameters"]}
    order = {p["name"]: i for i, p in enumerate(info["parameters"])}
    ordered = sorted(set(chosen.values()), key=lambda nm: order[nm])
    plist = ",".join(ordered)
    r = requests.get(f"{SERVER}/data", params={
        "dataset": DATASET, "parameters": plist,
        "start": f"{start}T00:00:00Z", "stop": f"{stop}T00:00:00Z",
        "format": "csv"}, timeout=600)
    r.raise_for_status()
    raw = pd.read_csv(io.StringIO(r.text), header=None, names=["time"] + ordered)
    raw["time"] = pd.to_datetime(raw["time"], utc=True).dt.tz_convert(None)
    raw = raw.set_index("time")
    out = pd.DataFrame(index=raw.index)
    inv = {chosen[c]: c for c in WANT}          # param name -> our column
    for pname in ordered:
        col = inv[pname]
        vals = raw[pname].astype("float64")
        f = fills.get(pname)
        if f not in (None, "null"):
            try:
                vals = vals.mask(np.isclose(vals, float(f), rtol=1e-3))
            except (TypeError, ValueError):
                pass
        vals = vals.mask(vals > 9e4)
        out[col] = vals
    return out[["bt", "bz", "by", "v", "n", "dst"]]


def fetch(start="2016-01-01", stop="2024-01-01", max_gap_hours=6) -> pd.DataFrame:
    """Fetch OMNI hourly solar wind + Dst -> tidy [bt,bz,by,v,n,dst] frame.

    Short gaps (<= max_gap_hours) are interpolated; longer gaps dropped.
    """
    try:
        df = _fetch_hapiclient(start, stop)
    except ImportError:
        df = _fetch_csv(start, stop)
    df = df.asfreq("h").interpolate(limit=max_gap_hours, limit_direction="both")
    df = df.dropna(subset=["bz", "v", "n", "dst"])
    return df[["bt", "bz", "by", "v", "n", "dst"]]
