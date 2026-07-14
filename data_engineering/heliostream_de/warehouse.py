"""DuckDB warehouse access: schema bootstrap and idempotent raw upserts."""
from __future__ import annotations

from typing import Iterable

import duckdb

from . import config as C


def connect(path=None, read_only=False):
    return duckdb.connect(str(path or C.DUCKDB_PATH), read_only=read_only)


def init_raw(con):
    """Create the raw landing table if absent. `time` is the natural key."""
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {C.RAW_SCHEMA};")
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {C.RAW_SCHEMA}.{C.RAW_TABLE} (
            time        TIMESTAMP PRIMARY KEY,
            bt          DOUBLE,
            bz          DOUBLE,
            by_gsm      DOUBLE,
            v           DOUBLE,
            n           DOUBLE,
            dst         DOUBLE,
            source      VARCHAR,
            ingested_at TIMESTAMP DEFAULT now()
        );
    """)


def upsert_records(con, records: Iterable[dict]) -> int:
    """Insert-or-replace records keyed on `time`. Returns rows written.

    Last-write-wins on duplicate timestamps (matches a Kafka replay / late
    correction landing after an earlier value).
    """
    rows = [
        (r.get("time"), r.get("bt"), r.get("bz"), r.get("by"),
         r.get("v"), r.get("n"), r.get("dst"), r.get("source"))
        for r in records
    ]
    if not rows:
        return 0
    con.executemany(f"""
        INSERT INTO {C.RAW_SCHEMA}.{C.RAW_TABLE}
            (time, bt, bz, by_gsm, v, n, dst, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (time) DO UPDATE SET
            bt=excluded.bt, bz=excluded.bz, by_gsm=excluded.by_gsm, v=excluded.v,
            n=excluded.n, dst=excluded.dst, source=excluded.source,
            ingested_at=now();
    """, rows)
    return len(rows)


def raw_stats(con) -> dict:
    init_raw(con)
    row = con.execute(f"""
        SELECT count(*) n, min(time) t0, max(time) t1,
               sum(CASE WHEN dst < -50 THEN 1 ELSE 0 END) storms
        FROM {C.RAW_SCHEMA}.{C.RAW_TABLE};
    """).fetchone()
    return {"rows": row[0], "min_time": row[1], "max_time": row[2], "storm_hours": row[3]}
