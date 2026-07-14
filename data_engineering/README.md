# Heliostream data engineering

A streaming plus batch pipeline that lands upstream solar-wind measurements in a
warehouse, transforms them into a tested feature mart with dbt, and feeds the
Heliostream model from the warehouse instead of an in-memory DataFrame.

This is the layer the notebook version was missing: real ingestion, storage,
transformation, tests, quality gates, and orchestration, rather than a single
`fetch()` call.

```
 source              bus                consumer         warehouse (DuckDB)         dbt                model
 ------              ---                --------         ------------------         ---                -----
 NOAA live  ─┐                       ┌─ validate ─┐                          ┌ stg_solar_wind (view)
 OMNI batch ─┼─▶ produce ─▶ topic ──┤            ├─▶ raw.solar_wind ──▶ dbt ─┤                       ─▶ features_hourly ─▶ train
 synthetic  ─┘   (Kafka /            └─ upsert ───┘   (idempotent,           └ tests + freshness +      (physics features,      (off the
                  file log)                            last-write-wins)         quality gate              parity-checked)          warehouse)
```

- **Bus**: Kafka/Redpanda for real; a file-backed log with a durable consumer
  offset for offline runs and tests. Same `produce`/`consume` interface.
- **Warehouse**: DuckDB, `raw` schema landed by the consumer, `main` schema
  built by dbt.
- **Transform**: dbt models (`stg_solar_wind` view, `features_hourly` table) with
  16 tests, source freshness, and singular tests for physical invariants.
- **Model**: trains directly off `features_hourly` via
  `heliostream.pipeline.load_dataset("warehouse", ...)`.

## Quickstart, offline (no Docker, no broker, no internet)

```bash
pip install -r requirements.txt && pip install -e ..   # model pkg for training
make demo
```

`make demo` runs the whole DAG (`produce -> load -> dbt build+test -> quality`)
against the file bus and a synthetic feed, then trains the model off the
resulting feature mart. Individual steps:

```bash
make batch      # produce -> load -> dbt build + test -> quality gate
make dbt        # just the transforms + tests
make quality    # warehouse data-quality checks
make train      # train the hybrid model off features_hourly
make stats      # raw landing stats
make test       # pipeline unit tests (bus, warehouse, feature parity)
```

## Quickstart, streaming (Docker + Redpanda)

```bash
make docker-up          # Redpanda + console + producer + consumer
#   producer streams synthetic hours -> Kafka; consumer lands them in DuckDB
make docker-dbt         # transform + test inside the stack
open http://localhost:8080   # Redpanda console (topics, messages, lag)
make docker-down
```

To use the real feed, change the `producer` command in `docker-compose.yml` from
`stream-synthetic` to `stream-noaa` (needs internet). Historical Dst is backfilled
separately by the OMNI batch source, mirroring how real operations get a
definitive Dst only after the fact.

## What it demonstrates

- **Streaming ingestion** with a swappable bus and an idempotent, last-write-wins
  landing table (so replays and late corrections are safe).
- **Warehouse modelling**: raw -> staging -> feature mart, materialized and
  ordered, with the transform logic in SQL.
- **dbt tests**: not-null / unique / accepted-values, source freshness thresholds,
  and singular tests for physics invariants (`|Bz| <= Bt`, `pdyn >= 0`).
- **A quality gate** the model refuses to train behind (null rates, value ranges,
  hourly-gap continuity).
- **Feature parity**: an automated test proves the dbt SQL features equal the
  pandas `engineer()` the model was validated against, so warehouse training is
  equivalent, not merely similar.
- **Orchestration**: a fail-fast DAG runner; each step maps cleanly to one
  Airflow/Dagster task. A **Dagster** project is included in `orchestration/`,
  where the dbt models load as native assets (with tests as asset checks) and an
  hourly schedule drives ingest -> transform -> quality.

## Honest limitations (say these in the writeup)

- The offline path uses a file-backed bus, not a real broker. The Kafka path is
  wired and runs under Docker, but the sandbox this was built in could not run a
  broker, so the Kafka path is validated by the compose config and code, not an
  executed broker test here.
- DuckDB is single-writer. The consumer releases its connection between drains and
  dbt runs on demand, so they do not collide, but true concurrent writers would
  need Postgres or similar. This is a deliberate laptop-friendly choice.
- Orchestration ships in two forms: a lightweight fail-fast runner
  (`heliostream_de run-batch`) and a full **Dagster** project (`orchestration/`)
  with dbt assets, asset checks, and an hourly schedule. The DAG boundaries are
  the same in both.
- The `stream-noaa` real feed and OMNI backfill need internet and are validated on
  a networked machine (see the model-side notebook for the OMNI path).
