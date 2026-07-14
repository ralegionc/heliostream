"""Custom Dagster assets wrapping the pipeline steps.

Lineage:  raw.solar_wind -> (dbt: stg_solar_wind -> features_hourly) ->
          quality_report -> trained_model

Run parameters are read from environment variables (kept simple and
version-robust):
  HELIO_INGEST_SOURCE  synthetic|noaa|omni   (default synthetic)
  HELIO_INGEST_HOURS   synthetic backfill length (default 8760)
  HELIO_TRAIN_MODEL    hybrid|gru            (default hybrid)
  HELIO_TRAIN_EPOCHS   default 40
"""
from __future__ import annotations

import os

from dagster import asset, AssetKey, MaterializeResult, MetadataValue

from heliostream_de import producer, consumer, quality, config as C
from heliostream_de import warehouse as W


@asset(
    key=AssetKey(["raw", "solar_wind"]),   # matches the dbt source key -> dbt runs downstream
    group_name="ingestion",
    compute_kind="python",
    description="Publish upstream solar wind to the bus and land it in DuckDB raw.solar_wind.",
)
def raw_solar_wind(context) -> MaterializeResult:
    source = os.environ.get("HELIO_INGEST_SOURCE", "synthetic")
    if source == "synthetic":
        hours = int(os.environ.get("HELIO_INGEST_HOURS", "8760"))
        published = producer.produce(source="synthetic", hours=hours)
    else:
        published = producer.produce(source=source)
    landed = consumer.drain_once()
    stats = W.raw_stats(W.connect())
    context.log.info(f"published={published} landed={landed} total={stats['rows']}")
    return MaterializeResult(metadata={
        "records_published": published,
        "rows_landed": landed,
        "total_rows": stats["rows"],
        "storm_hours": stats["storm_hours"],
        "latest_hour": str(stats["max_time"]),
        "source": source,
    })


@asset(
    deps=[AssetKey("features_hourly")],    # dbt model
    group_name="quality",
    compute_kind="python",
    description="Warehouse data-quality gate. Raises (fails the run) if any check fails.",
)
def quality_report(context) -> MaterializeResult:
    rep = quality.run_checks()
    for c in rep["checks"]:
        context.log.info(f"[{'ok' if c['pass'] else 'FAIL'}] {c['check']}: {c['detail']}")
    if not rep["passed"]:
        failed = [c["check"] for c in rep["checks"] if not c["pass"]]
        raise Exception(f"quality gate failed: {failed}")
    return MaterializeResult(metadata={
        "rows": rep["n"],
        "latest": rep["latest"],
        "passed": rep["passed"],
        "checks": MetadataValue.json(rep["checks"]),
    })


@asset(
    deps=[AssetKey("features_hourly"), AssetKey("quality_report")],
    group_name="model",
    compute_kind="pytorch",
    description="Train the Heliostream model directly off the warehouse feature mart.",
)
def trained_model(context) -> MaterializeResult:
    from heliostream import train as T, evaluate as E
    from heliostream.data import features as F
    from heliostream.pipeline import load_dataset

    model_name = os.environ.get("HELIO_TRAIN_MODEL", "hybrid")
    epochs = int(os.environ.get("HELIO_TRAIN_EPOCHS", "40"))
    df = load_dataset("warehouse", duckdb_path=C.DUCKDB_PATH)
    context.log.info(f"training {model_name} on {len(df)} hours from features_hourly")
    model, norm, hist = T.train_model(df, model_name, epochs=epochs, verbose=False)
    T.save(model, norm, model_name, hist, extra={"source": "warehouse"})
    dfe = F.engineer(df); _, va, te = F.time_split(dfe)
    rep, (m, s, y, factors) = E.full_report(model, norm, va, te)
    E.save_calibration(factors)
    return MaterializeResult(metadata={
        "model": model_name,
        "train_hours": len(df),
        "rmse_nt": round(rep["model"]["rmse_all"], 2),
        "storm_rmse_nt": round(rep["model"]["rmse_storm"], 2) if rep["model"]["rmse_storm"] else None,
        "coverage_90": round(rep["conformal"]["coverage_conformal"], 3),
    })
