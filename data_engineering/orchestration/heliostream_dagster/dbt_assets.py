"""dbt <-> Dagster bridge.

Loads the Heliostream dbt project so every model (stg_solar_wind, features_hourly)
and its tests appear as native Dagster assets, wired downstream of the
`raw.solar_wind` source asset the ingestion step materializes.
"""
from __future__ import annotations

import os
import subprocess

from dagster import AssetExecutionContext
from dagster_dbt import DbtProject, DbtCliResource, dbt_assets

from heliostream_de import config as C

DBT_DIR = (C.ROOT / "dbt").resolve()

# Point dbt (and everything else) at the same DuckDB file.
os.environ.setdefault("HELIO_DUCKDB", str(C.DUCKDB_PATH))

dbt_project = DbtProject(project_dir=os.fspath(DBT_DIR), profiles_dir=os.fspath(DBT_DIR))
dbt_project.prepare_if_dev()

# Ensure a manifest exists even outside `dagster dev` (e.g. tests / webserver).
if not dbt_project.manifest_path.exists():
    subprocess.run(
        ["dbt", "parse", "--project-dir", os.fspath(DBT_DIR),
         "--profiles-dir", os.fspath(DBT_DIR)],
        env={**os.environ, "HELIO_DUCKDB": str(C.DUCKDB_PATH)},
        check=True,
    )


@dbt_assets(manifest=dbt_project.manifest_path)
def heliostream_dbt_assets(context, dbt: DbtCliResource):
    """Run `dbt build` (models + tests) and stream results back to Dagster."""
    yield from dbt.cli(["build"], context=context).stream()
