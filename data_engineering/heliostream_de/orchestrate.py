"""Batch orchestration.

Runs the pipeline DAG in order and fails fast on any step:

    backfill (source -> bus) -> load (bus -> raw) -> dbt build (transform + test)
    -> quality gate -> ready for training.

Deliberately a small, dependency-free runner. Wrapping these same steps in
Airflow/Dagster is the labelled next step; each function here maps to one task.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from . import config as C
from . import producer, consumer, quality


def _log(step, msg=""):
    print(f"[orchestrate] {step:<14} {msg}")


def step_backfill(source="synthetic", **kw):
    _log("backfill", f"source={source}")
    n = producer.produce(source=source, **kw)
    _log("backfill", f"published {n} records to bus")
    return n


def step_load():
    _log("load", "draining bus -> raw.solar_wind")
    n = consumer.drain_once()
    _log("load", f"landed {n} rows")
    return n


def step_dbt(command="build"):
    dbt_dir = C.ROOT / "dbt"
    env = os.environ.copy()
    env["HELIO_DUCKDB"] = str(C.DUCKDB_PATH)
    _log("dbt", f"dbt {command} (profiles in {dbt_dir})")
    r = subprocess.run(
        ["dbt", command, "--project-dir", str(dbt_dir), "--profiles-dir", str(dbt_dir)],
        env=env, capture_output=True, text=True,
    )
    print(r.stdout[-4000:])
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise RuntimeError(f"dbt {command} failed (exit {r.returncode})")
    return r.returncode


def step_quality():
    _log("quality", "running warehouse checks")
    rep = quality.run_checks()
    for c in rep["checks"]:
        mark = "ok " if c["pass"] else "FAIL"
        print(f"    [{mark}] {c['check']}: {c['detail']}")
    if not rep["passed"]:
        raise RuntimeError("quality gate failed")
    _log("quality", f"passed ({rep['n']} rows, latest {rep['latest']})")
    return rep


def run_batch(source="synthetic", dbt_command="build", **kw):
    t0 = time.time()
    step_backfill(source=source, **kw)
    step_load()
    step_dbt(dbt_command)
    rep = step_quality()
    _log("done", f"pipeline green in {time.time()-t0:.1f}s")
    return rep
