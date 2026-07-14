"""Heliostream orchestration (Dagster).

Wraps the data-engineering DAG as Dagster assets with dbt lineage:
    raw.solar_wind -> stg_solar_wind -> features_hourly -> quality_report -> trained_model

Run:  cd data_engineering/orchestration && dagster dev
"""
from .definitions import defs

__all__ = ["defs"]
