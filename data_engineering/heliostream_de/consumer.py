"""Consumer: drain the bus and land validated records into raw.solar_wind."""
from __future__ import annotations

import time

from . import config as C
from . import warehouse as W
from .bus import get_bus


def _valid(rec: dict) -> bool:
    """Cheap gate before landing: must have a timestamp and sane magnitudes."""
    if not rec.get("time"):
        return False
    for k in ("v", "n", "bt"):
        x = rec.get(k)
        if x is not None and (x != x or x > 9e4):   # NaN or fill
            return False
    return True


def drain_once(bus_backend=None, batch=10_000) -> int:
    """Consume available records and upsert them. Returns rows landed."""
    bus = get_bus(bus_backend)
    con = W.connect()
    W.init_raw(con)
    total = 0
    while True:
        recs = bus.consume(max_records=batch)
        if not recs:
            break
        good = [r for r in recs if _valid(r)]
        total += W.upsert_records(con, good)
        if len(recs) < batch:
            break
    con.close()
    return total


def run(bus_backend=None, interval_s=30, once=False):
    """Long-running loader: repeatedly drain the bus into the warehouse."""
    while True:
        landed = drain_once(bus_backend)
        if landed:
            print(f"[consumer] landed {landed} rows into "
                  f"{C.RAW_SCHEMA}.{C.RAW_TABLE}")
        if once:
            return landed
        time.sleep(interval_s)
