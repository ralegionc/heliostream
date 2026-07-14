"""Heliostream data-engineering CLI.

Examples
--------
  python -m heliostream_de run-batch --source synthetic --hours 26280   # full DAG
  python -m heliostream_de produce --source synthetic --hours 8760
  python -m heliostream_de consume
  python -m heliostream_de dbt build
  python -m heliostream_de quality
  python -m heliostream_de stats
  python -m heliostream_de train --model hybrid --epochs 40             # off warehouse
  python -m heliostream_de stream-noaa   # live producer (needs Kafka + internet)
"""
from __future__ import annotations

import argparse
import json

from . import config as C
from . import producer, consumer, quality, orchestrate
from . import warehouse as W


def cmd_produce(a):
    kw = {"hours": a.hours, "seed": a.seed} if a.source == "synthetic" else {}
    if a.source == "omni":
        kw = {"start": a.start, "stop": a.stop}
    n = producer.produce(source=a.source, **kw)
    print(f"published {n} records to bus ({C.BUS_BACKEND})")


def cmd_consume(a):
    n = consumer.drain_once()
    print(f"landed {n} rows into {C.RAW_SCHEMA}.{C.RAW_TABLE}")


def cmd_stream_noaa(a):
    print("streaming NOAA -> bus (Ctrl-C to stop)")
    producer.stream_noaa(interval_s=a.interval, once=a.once)


def cmd_stream_synthetic(a):
    print("streaming synthetic -> bus (Ctrl-C to stop)")
    producer.stream_synthetic(interval_s=a.interval, seed=a.seed, once=a.once)


def cmd_consume_loop(a):
    consumer.run(interval_s=a.interval, once=a.once)


def cmd_dbt(a):
    orchestrate.step_dbt(a.command)


def cmd_quality(a):
    rep = quality.run_checks()
    print(json.dumps(rep, indent=2, default=str))
    raise SystemExit(0 if rep["passed"] else 1)


def cmd_stats(a):
    con = W.connect()
    print(json.dumps(W.raw_stats(con), indent=2, default=str))


def cmd_run_batch(a):
    kw = {"hours": a.hours, "seed": a.seed} if a.source == "synthetic" else {}
    if a.source == "omni":
        kw = {"start": a.start, "stop": a.stop}
    orchestrate.run_batch(source=a.source, **kw)


def cmd_train(a):
    # Train the Heliostream model directly off the warehouse feature mart.
    from heliostream import train as T
    from heliostream import evaluate as E
    from heliostream.data import features as F
    from heliostream.pipeline import load_dataset
    df = load_dataset("warehouse", duckdb_path=C.DUCKDB_PATH)
    print(f"loaded {len(df)} hours from warehouse feature mart")
    model, norm, hist = T.train_model(df, a.model, epochs=a.epochs)
    T.save(model, norm, a.model, hist, extra={"source": "warehouse"})
    dfe = F.engineer(df); _, va, te = F.time_split(dfe)
    rep, (m, s, y, factors) = E.full_report(model, norm, va, te)
    E.save_calibration(factors)
    print(f"test RMSE={rep['model']['rmse_all']:.2f}nT  "
          f"storm={rep['model']['rmse_storm']:.2f}nT  "
          f"coverage@90={rep['conformal']['coverage_conformal']:.3f}")


def build_parser():
    p = argparse.ArgumentParser(prog="heliostream_de")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_source(sp):
        sp.add_argument("--source", choices=["synthetic", "noaa", "omni"], default="synthetic")
        sp.add_argument("--hours", type=int, default=24 * 365 * 3)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--start", default="2016-01-01")
        sp.add_argument("--stop", default="2024-01-01")

    s = sub.add_parser("produce"); add_source(s); s.set_defaults(fn=cmd_produce)
    s = sub.add_parser("consume"); s.set_defaults(fn=cmd_consume)
    s = sub.add_parser("stream-noaa")
    s.add_argument("--interval", type=int, default=3600); s.add_argument("--once", action="store_true")
    s.set_defaults(fn=cmd_stream_noaa)
    s = sub.add_parser("stream-synthetic")
    s.add_argument("--interval", type=int, default=5); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--once", action="store_true"); s.set_defaults(fn=cmd_stream_synthetic)
    s = sub.add_parser("consume-loop")
    s.add_argument("--interval", type=int, default=15); s.add_argument("--once", action="store_true")
    s.set_defaults(fn=cmd_consume_loop)
    s = sub.add_parser("dbt"); s.add_argument("command", nargs="?", default="build")
    s.set_defaults(fn=cmd_dbt)
    s = sub.add_parser("quality"); s.set_defaults(fn=cmd_quality)
    s = sub.add_parser("stats"); s.set_defaults(fn=cmd_stats)
    s = sub.add_parser("run-batch"); add_source(s); s.set_defaults(fn=cmd_run_batch)
    s = sub.add_parser("train")
    s.add_argument("--model", choices=["hybrid", "gru"], default="hybrid")
    s.add_argument("--epochs", type=int, default=40); s.set_defaults(fn=cmd_train)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
