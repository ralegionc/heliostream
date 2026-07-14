"""Producer: read from a source and publish records onto the bus."""
from __future__ import annotations

import time

from . import config as C
from . import sources
from .bus import get_bus

SOURCES = {
    "synthetic": sources.synthetic_records,
    "noaa": sources.noaa_records,
    "omni": sources.omni_backfill_records,
}


def produce(source="synthetic", bus_backend=None, **source_kw):
    """Publish all records from `source` to the bus. Returns count."""
    bus = get_bus(bus_backend)
    n = 0
    for rec in SOURCES[source](**source_kw):
        bus.produce(rec)
        n += 1
    bus.flush()
    return n


def stream_synthetic(interval_s=5, seed=0, bus_backend=None, once=False, batch=1):
    """Emit synthetic hours to the bus on an interval (offline/compose demo)."""
    from heliostream.data import synthetic
    df = synthetic.simulate(24 * 400, seed=seed)
    recs = list(sources._frame_to_records(df, "synthetic"))
    bus = get_bus(bus_backend)
    i = 0
    while True:
        for _ in range(batch):
            bus.produce(recs[i % len(recs)])
            i += 1
        bus.flush()
        print(f"[producer] emitted {i} synthetic records")
        if once:
            break
        time.sleep(interval_s)


def stream_noaa(interval_s=3600, bus_backend=None, once=False):
    """Poll NOAA on an interval and publish new hours (a long-running service)."""
    bus = get_bus(bus_backend)
    seen = set()
    while True:
        try:
            for rec in sources.noaa_records():
                if rec["time"] not in seen:
                    bus.produce(rec)
                    seen.add(rec["time"])
            bus.flush()
        except Exception as e:  # keep the daemon alive across transient feed errors
            print(f"[producer] fetch error: {e}")
        if once:
            break
        time.sleep(interval_s)
