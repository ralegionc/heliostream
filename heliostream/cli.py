"""Heliostream command-line interface.

Examples
--------
  python -m heliostream train --source synthetic --model hybrid --epochs 40
  python -m heliostream evaluate --model hybrid
  python -m heliostream backtest --model hybrid --folds 4
  python -m heliostream serve --demo
  python -m heliostream train --source omni --start 2016-01-01 --stop 2024-01-01
"""
from __future__ import annotations

import argparse
import json
import warnings

warnings.filterwarnings("ignore")

from . import config as C
from . import train as T
from . import evaluate as E
from . import backtest as B
from .data import features as F
from .pipeline import load_dataset


def cmd_simulate(a):
    df = load_dataset("synthetic", refresh=True, years=a.years)
    print(f"simulated {len(df)} hours -> {C.DATA_DIR/'synthetic.parquet'}")
    print(f"storm hours (Dst<{C.STORM_THRESHOLD:.0f}): {(df.dst<C.STORM_THRESHOLD).sum()}")


def cmd_train(a):
    df = load_dataset(a.source, refresh=a.refresh, years=a.years,
                      start=a.start, stop=a.stop)
    print(f"loaded {len(df)} hours from '{a.source}'")
    model, norm, hist = T.train_model(df, a.model, epochs=a.epochs,
                                      storm_weight=a.storm_weight, hidden=a.hidden)
    T.save(model, norm, a.model, hist, extra={"source": a.source})
    print(f"saved model -> {C.MODEL_DIR/(a.model+'.pt')}")
    # auto-evaluate + calibrate so serving works immediately
    dfe = F.engineer(df); _, va, te = F.time_split(dfe)
    rep, (mean, std, y, factors) = E.full_report(model, norm, va, te)
    E.save_calibration(factors, alpha=0.1)
    (C.REPORT_DIR / f"{a.model}_report.json").write_text(json.dumps(rep, indent=2, default=str))
    _print_report(rep)


def cmd_evaluate(a):
    model, norm, meta = T.load(a.model)
    df = load_dataset(meta.get("source", "synthetic"))
    dfe = F.engineer(df); _, va, te = F.time_split(dfe)
    rep, (mean, std, y, factors) = E.full_report(model, norm, va, te)
    E.save_calibration(factors, alpha=0.1)
    p1, p2 = E.save_plots(model, norm, te, factors, prefix=a.model)
    (C.REPORT_DIR / f"{a.model}_report.json").write_text(json.dumps(rep, indent=2, default=str))
    _print_report(rep)
    print(f"plots -> {p1}\n         {p2}")


def cmd_backtest(a):
    df = load_dataset(a.source, years=a.years)
    res = B.walk_forward(df, a.model, folds=a.folds, epochs=a.epochs)
    (C.REPORT_DIR / f"{a.model}_backtest.json").write_text(json.dumps(res, indent=2, default=str))
    rmse = [r["rmse_all"] for r in res]
    print(f"\nwalk-forward mean rmse={sum(rmse)/len(rmse):.2f}nT over {len(res)} folds")


def cmd_serve(a):
    from . import serve
    print(f"serving http://{a.host}:{a.port}  (demo={a.demo})")
    serve.run(a.model, demo=a.demo, host=a.host, port=a.port)


def _print_report(rep):
    m, pe, ob = rep["model"], rep["persistence"], rep["obm_empirical"]
    cf = rep["conformal"]
    print("\n=== test metrics ===")
    print(f"{'model':<14}{'RMSE':>8}{'storm-RMSE':>12}{'MAE':>8}")
    for name, r in [("hybrid/gru", m), ("persistence", pe), ("obm-empirical", ob)]:
        sr = f"{r['rmse_storm']:.2f}" if r['rmse_storm'] else "n/a"
        print(f"{name:<14}{r['rmse_all']:>8.2f}{sr:>12}{r['mae_all']:>8.2f}")
    print(f"\ncoverage@90  gaussian={cf['coverage_gaussian']:.3f}  "
          f"conformal={cf['coverage_conformal']:.3f}  (target {cf['target_coverage']})")


def build_parser():
    p = argparse.ArgumentParser(prog="heliostream",
                                description="Physics-informed geomagnetic storm nowcasting")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="generate synthetic dataset")
    s.add_argument("--years", type=float, default=6); s.set_defaults(fn=cmd_simulate)

    s = sub.add_parser("train", help="train a model")
    s.add_argument("--source", choices=["synthetic", "omni"], default="synthetic")
    s.add_argument("--model", choices=["hybrid", "gru"], default="hybrid")
    s.add_argument("--epochs", type=int, default=40)
    s.add_argument("--hidden", type=int, default=64)
    s.add_argument("--storm-weight", dest="storm_weight", type=float, default=1.0)
    s.add_argument("--years", type=float, default=6)
    s.add_argument("--start", default="2016-01-01"); s.add_argument("--stop", default="2024-01-01")
    s.add_argument("--refresh", action="store_true"); s.set_defaults(fn=cmd_train)

    s = sub.add_parser("evaluate", help="evaluate + calibrate + plots")
    s.add_argument("--model", choices=["hybrid", "gru"], default="hybrid")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("backtest", help="walk-forward backtest")
    s.add_argument("--model", choices=["hybrid", "gru"], default="hybrid")
    s.add_argument("--source", choices=["synthetic", "omni"], default="synthetic")
    s.add_argument("--folds", type=int, default=4); s.add_argument("--epochs", type=int, default=25)
    s.add_argument("--years", type=float, default=6); s.set_defaults(fn=cmd_backtest)

    s = sub.add_parser("serve", help="live dashboard + API")
    s.add_argument("--model", choices=["hybrid", "gru"], default="hybrid")
    s.add_argument("--demo", action="store_true")
    s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
