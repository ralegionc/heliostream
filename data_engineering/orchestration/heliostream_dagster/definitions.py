"""Top-level Dagster definitions: assets, a pipeline job, an hourly schedule,
and the dbt resource. Loaded by `dagster dev`.
"""
from __future__ import annotations

from dagster import (
    Definitions, define_asset_job, AssetSelection, ScheduleDefinition,
)
from dagster_dbt import DbtCliResource

from .dbt_assets import dbt_project, heliostream_dbt_assets
from .assets import raw_solar_wind, quality_report, trained_model

# Ingestion + transform + quality (everything except the heavy training step).
pipeline_job = define_asset_job(
    name="heliostream_pipeline",
    selection=AssetSelection.all() - AssetSelection.assets(trained_model),
)

# The upstream solar wind arrives hourly, so refresh on the hour.
hourly_schedule = ScheduleDefinition(
    name="hourly_ingest",
    job=pipeline_job,
    cron_schedule="0 * * * *",
)

# Full graph including training, for on-demand runs.
full_job = define_asset_job(name="heliostream_pipeline_with_training",
                            selection=AssetSelection.all())

defs = Definitions(
    assets=[raw_solar_wind, heliostream_dbt_assets, quality_report, trained_model],
    jobs=[pipeline_job, full_job],
    schedules=[hourly_schedule],
    resources={"dbt": DbtCliResource(project_dir=dbt_project)},
)
