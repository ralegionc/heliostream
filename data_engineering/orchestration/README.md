# Heliostream orchestration (Dagster)

Wraps the pipeline as Dagster assets, with the dbt models loaded as **native
assets** (via `dagster-dbt`) so the whole DAG and its lineage render in the
Dagster UI, and every dbt test shows up as a Dagster **asset check**.

```
raw/solar_wind ─▶ stg_solar_wind ─▶ features_hourly ─▶ quality_report ─▶ trained_model
   (ingestion)        (dbt)              (dbt)            (quality gate)     (pytorch)
```

- `raw/solar_wind` — publishes to the bus and lands rows in `raw.solar_wind`
  (its asset key matches the dbt source, so dbt runs downstream automatically).
- `stg_solar_wind`, `features_hourly` — the dbt models, with `not_null` / `unique`
  / `accepted_values` and the singular physics tests attached as asset checks.
- `quality_report` — the warehouse quality gate; fails the run if a check fails.
- `trained_model` — trains the Heliostream model off `features_hourly` (excluded
  from the scheduled job; run on demand).

Two jobs are defined: `heliostream_pipeline` (ingest → transform → quality) and
`heliostream_pipeline_with_training` (adds the model). An `hourly_ingest`
schedule runs the former on the hour, matching the upstream cadence.

## Run

```bash
cd data_engineering/orchestration
pip install -e .            # dagster, dagster-dbt, dagster-webserver
pip install -e ../..        # the heliostream model package (for the training asset)
export PYTHONPATH=..:.      # so heliostream_de and heliostream_dagster import
dagster dev                 # UI at http://localhost:3000
```

In the UI: open Assets, click Materialize to run the graph, and watch the dbt
models, asset checks, and quality gate execute with live logs and metadata
(rows landed, storm hours, coverage). Or run headless:

```bash
dagster job execute -m heliostream_dagster -j heliostream_pipeline
```

## Config (environment variables)

- `HELIO_INGEST_SOURCE` — `synthetic` (default) | `noaa` | `omni`
- `HELIO_INGEST_HOURS` — synthetic backfill length (default 8760)
- `HELIO_TRAIN_MODEL` / `HELIO_TRAIN_EPOCHS` — for the `trained_model` asset
- `HELIO_DUCKDB` — warehouse path (defaults to `../warehouse/heliostream.duckdb`)

## Notes / honest limits

- The dbt models appear as assets by loading the project's manifest; it is
  generated automatically on first import (`dbt parse`) if missing.
- DuckDB is single-writer, so keep `dagster dev` from materializing while the
  Compose consumer is also writing. For concurrent writers, move to Postgres.
- This was validated by executing the `heliostream_pipeline` job in-process
  (ingest → dbt build with asset checks → quality gate, all green); the live
  `dagster dev` webserver is the same definitions behind a UI.
