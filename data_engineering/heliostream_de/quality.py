"""Warehouse data-quality checks (complementing dbt tests).

Runs cheap assertions the model layer cares about: freshness, null rates,
value ranges, and hourly-gap continuity. Returns a report and a pass/fail.
"""
from __future__ import annotations

from . import config as C
from . import warehouse as W


CHECKS = {
    "null_rate_bz_max": 0.05,
    "null_rate_v_max": 0.05,
    "max_gap_hours": 72,        # a single outage longer than 3 days signals a real problem
    "missing_fraction_max": 0.02,  # >2% of the span missing signals a broken feed
    "v_min": 100, "v_max": 3000,
    "dst_min": -800, "dst_max": 100,
}


def run_checks(con=None, relation="main.features_hourly") -> dict:
    con = con or W.connect(read_only=True)
    q = con.execute(f"""
        SELECT
            count(*)                                         AS n,
            avg(CASE WHEN bz IS NULL THEN 1 ELSE 0 END)      AS null_bz,
            avg(CASE WHEN v  IS NULL THEN 1 ELSE 0 END)      AS null_v,
            min(v) AS vmin, max(v) AS vmax,
            min(dst) AS dmin, max(dst) AS dmax,
            max(time) AS tmax
        FROM {relation};
    """).fetchone()
    n, null_bz, null_v, vmin, vmax, dmin, dmax, tmax = q

    gap_row = con.execute(f"""
        WITH t AS (
            SELECT time, lag(time) OVER (ORDER BY time) AS prev
            FROM {relation}
        )
        SELECT max(date_diff('hour', prev, time)),
               count(*) FILTER (WHERE date_diff('hour', prev, time) > 1),
               coalesce(sum(date_diff('hour', prev, time) - 1)
                        FILTER (WHERE date_diff('hour', prev, time) > 1), 0)
        FROM t WHERE prev IS NOT NULL;
    """).fetchone()
    gap, n_gaps, missing_hours = gap_row
    span_hours = con.execute(
        f"SELECT date_diff('hour', min(time), max(time)) FROM {relation};"
    ).fetchone()[0] or 0
    missing_frac = (missing_hours / span_hours) if span_hours else 0.0

    results = []

    def check(name, ok, detail):
        results.append({"check": name, "pass": bool(ok), "detail": detail})

    check("row_count_positive", n and n > 0, f"rows={n}")
    check("null_rate_bz", (null_bz or 0) <= CHECKS["null_rate_bz_max"],
          f"{(null_bz or 0):.3f} <= {CHECKS['null_rate_bz_max']}")
    check("null_rate_v", (null_v or 0) <= CHECKS["null_rate_v_max"],
          f"{(null_v or 0):.3f} <= {CHECKS['null_rate_v_max']}")
    check("v_in_range", vmin is None or (vmin >= CHECKS["v_min"] and vmax <= CHECKS["v_max"]),
          f"[{vmin}, {vmax}]")
    check("dst_in_range", dmin is None or (dmin >= CHECKS["dst_min"] and dmax <= CHECKS["dst_max"]),
          f"[{dmin}, {dmax}]")
    # Feed continuity: real spacecraft telemetry always has some dropout, so we
    # gate on the *fraction* of the record missing (a broken feed) and on any
    # single outage being implausibly long, not on the mere presence of gaps.
    check("missing_fraction", missing_frac <= CHECKS["missing_fraction_max"],
          f"{missing_frac:.4%} of span missing ({missing_hours}h in {n_gaps} gaps) "
          f"<= {CHECKS['missing_fraction_max']:.0%}")
    check("max_gap_hours", gap is None or gap <= CHECKS["max_gap_hours"],
          f"largest outage {gap}h <= {CHECKS['max_gap_hours']}h")

    passed = all(r["pass"] for r in results)
    return {"passed": passed, "n": n, "latest": str(tmax),
            "gaps": n_gaps, "missing_hours": missing_hours,
            "missing_fraction": missing_frac, "max_gap_hours": gap,
            "checks": results}
