"""Data access for the warehouse dashboard.

Pure functions that query the DuckDB feature mart and return DataFrames / dicts.
Kept separate from the Streamlit UI so they can be unit-tested without a browser.
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd

RELATION = "main.features_hourly"
RAW = "raw.solar_wind"

SEVERITY = [
    ("Quiet", 0, "> -30 nT"),
    ("Unsettled", 1, "-30 to -50"),
    ("Minor (G1)", 2, "-50 to -100"),
    ("Moderate-strong (G2-G3)", 3, "-100 to -200"),
    ("Severe (G4-G5)", 4, "< -200"),
]


def default_db_path() -> Path:
    env = os.environ.get("HELIO_DUCKDB")
    if env:
        return Path(env)
    # dashboard/ -> data_engineering/ -> warehouse/heliostream.duckdb
    return Path(__file__).resolve().parent.parent / "warehouse" / "heliostream.duckdb"


def connect(path=None):
    return duckdb.connect(str(path or default_db_path()), read_only=True)


def has_features(con) -> bool:
    try:
        con.execute(f"SELECT 1 FROM {RELATION} LIMIT 1").fetchone()
        return True
    except Exception:
        return False


def kpis(con) -> dict:
    row = con.execute(f"""
        SELECT count(*) n, min(time) t0, max(time) t1,
               sum(CASE WHEN dst < -50 THEN 1 ELSE 0 END) storms,
               sum(CASE WHEN dst < -100 THEN 1 ELSE 0 END) intense,
               min(dst) dmin
        FROM {RELATION};
    """).fetchone()
    n, t0, t1, storms, intense, dmin = row
    latest = pd.Timestamp(t1) if t1 is not None else None
    span_days = (pd.Timestamp(t1) - pd.Timestamp(t0)).days if t0 and t1 else 0
    return {
        "rows": int(n or 0),
        "t0": t0, "t1": t1, "span_days": span_days,
        "storm_hours": int(storms or 0),
        "intense_hours": int(intense or 0),
        "min_dst": float(dmin) if dmin is not None else None,
        "latest": latest,
    }


def dst_daily(con) -> pd.DataFrame:
    """Daily storm depth (min Dst) and mean, for the timeline."""
    return con.execute(f"""
        SELECT date_trunc('day', time) AS day,
               min(dst) AS dst_min,
               avg(dst) AS dst_mean
        FROM {RELATION}
        GROUP BY 1 ORDER BY 1;
    """).df()


def severity_distribution(con) -> pd.DataFrame:
    df = con.execute(f"""
        SELECT
          CASE
            WHEN dst >= -30  THEN 'Quiet'
            WHEN dst >= -50  THEN 'Unsettled'
            WHEN dst >= -100 THEN 'Minor (G1)'
            WHEN dst >= -200 THEN 'Moderate-strong (G2-G3)'
            ELSE 'Severe (G4-G5)'
          END AS severity,
          count(*) AS hours
        FROM {RELATION} GROUP BY 1;
    """).df()
    order = {name: i for name, i, _ in SEVERITY}
    df["rank"] = df["severity"].map(order)
    return df.sort_values("rank").reset_index(drop=True)


def source_coverage(con) -> pd.DataFrame:
    return con.execute(f"""
        SELECT source, count(*) AS hours
        FROM {RELATION} GROUP BY 1 ORDER BY hours DESC;
    """).df()


def recent_solar_wind(con, hours=168) -> pd.DataFrame:
    return con.execute(f"""
        SELECT time, bz, bt, v, dst
        FROM {RELATION}
        ORDER BY time DESC LIMIT {int(hours)};
    """).df().sort_values("time")


def monthly_storm_counts(con) -> pd.DataFrame:
    return con.execute(f"""
        SELECT date_trunc('month', time) AS month,
               sum(CASE WHEN dst < -50 THEN 1 ELSE 0 END) AS storm_hours
        FROM {RELATION} GROUP BY 1 ORDER BY 1;
    """).df()


def gap_summary(con) -> dict:
    """Feed continuity: gap count, missing hours, largest outage, missing fraction."""
    gap, n_gaps, missing = con.execute(f"""
        WITH t AS (SELECT time, lag(time) OVER (ORDER BY time) prev FROM {RELATION})
        SELECT max(date_diff('hour', prev, time)),
               count(*) FILTER (WHERE date_diff('hour', prev, time) > 1),
               coalesce(sum(date_diff('hour', prev, time) - 1)
                        FILTER (WHERE date_diff('hour', prev, time) > 1), 0)
        FROM t WHERE prev IS NOT NULL;
    """).fetchone()
    span = con.execute(
        f"SELECT date_diff('hour', min(time), max(time)) FROM {RELATION};").fetchone()[0] or 0
    return {"max_gap": gap or 0, "n_gaps": n_gaps or 0,
            "missing_hours": missing or 0,
            "missing_fraction": (missing / span) if span else 0.0}


def largest_gaps(con, limit=8) -> pd.DataFrame:
    return con.execute(f"""
        WITH t AS (SELECT time, lag(time) OVER (ORDER BY time) prev FROM {RELATION})
        SELECT prev AS gap_start, time AS gap_end,
               date_diff('hour', prev, time) AS gap_hours
        FROM t WHERE date_diff('hour', prev, time) > 1
        ORDER BY gap_hours DESC LIMIT {int(limit)};
    """).df()


def quality_checks(con) -> pd.DataFrame:
    """Self-contained data-quality gate over the feature mart.

    Feed continuity is gated on the *fraction* of the record missing and on any
    single outage being implausibly long. Real spacecraft telemetry always has
    some dropout, so the presence of gaps is reported, not failed.
    """
    n, null_bz, null_v, vmin, vmax, dmin, dmax = con.execute(f"""
        SELECT count(*),
               avg(CASE WHEN bz IS NULL THEN 1 ELSE 0 END),
               avg(CASE WHEN v  IS NULL THEN 1 ELSE 0 END),
               min(v), max(v), min(dst), max(dst)
        FROM {RELATION};
    """).fetchone()
    g = gap_summary(con)

    rows = [
        ("rows present", (n or 0) > 0, f"{n:,}"),
        ("Bz null rate <= 5%", (null_bz or 0) <= 0.05, f"{(null_bz or 0):.1%}"),
        ("V null rate <= 5%", (null_v or 0) <= 0.05, f"{(null_v or 0):.1%}"),
        ("V in [100, 3000]", vmin is None or (vmin >= 100 and vmax <= 3000),
         f"[{vmin:.0f}, {vmax:.0f}]" if vmin is not None else "n/a"),
        ("Dst in [-800, 100]", dmin is None or (dmin >= -800 and dmax <= 100),
         f"[{dmin:.0f}, {dmax:.0f}]" if dmin is not None else "n/a"),
        ("Feed coverage >= 98%", g["missing_fraction"] <= 0.02,
         f"{1 - g['missing_fraction']:.2%} covered "
         f"({g['missing_hours']}h missing in {g['n_gaps']} gaps)"),
        ("No outage > 72 h", g["max_gap"] <= 72, f"largest {g['max_gap']} h"),
    ]
    return pd.DataFrame(rows, columns=["check", "pass", "detail"])
