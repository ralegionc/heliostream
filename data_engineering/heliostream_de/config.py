"""Pipeline configuration (env-overridable)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # data_engineering/
WAREHOUSE_DIR = Path(os.environ.get("HELIO_WAREHOUSE_DIR", ROOT / "warehouse"))
WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
DUCKDB_PATH = Path(os.environ.get("HELIO_DUCKDB", WAREHOUSE_DIR / "heliostream.duckdb"))

# message bus
BUS_BACKEND = os.environ.get("HELIO_BUS", "file")       # "kafka" | "file"
KAFKA_BROKERS = os.environ.get("HELIO_KAFKA_BROKERS", "localhost:19092")
TOPIC_RAW = os.environ.get("HELIO_TOPIC", "solarwind.raw")
FILE_BUS_PATH = Path(os.environ.get("HELIO_FILEBUS", WAREHOUSE_DIR / "bus" / f"{TOPIC_RAW}.log"))
CONSUMER_GROUP = os.environ.get("HELIO_GROUP", "heliostream-loader")

# warehouse layout
RAW_SCHEMA = "raw"
RAW_TABLE = "solar_wind"                                 # raw.solar_wind
FEATURES_RELATION = "features_hourly"                    # produced by dbt (main schema)

# canonical record fields landed from the bus
RECORD_FIELDS = ["time", "bt", "bz", "by", "v", "n", "dst", "source"]
