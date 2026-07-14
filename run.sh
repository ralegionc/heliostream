#!/usr/bin/env bash
# End-to-end offline demo: install, train, evaluate, and launch the dashboard.
# No internet or GPU required (uses the synthetic simulator).
set -e
cd "$(dirname "$0")"

python -m pip install -r requirements.txt

echo "== training physics-informed hybrid (synthetic, CPU) =="
python -m heliostream train --source synthetic --model hybrid --epochs 35

echo "== training black-box GRU baseline for comparison =="
python -m heliostream train --source synthetic --model gru --epochs 35

echo "== evaluate + calibrate + plots =="
python -m heliostream evaluate --model hybrid

echo "== launching live dashboard (demo feed) at http://127.0.0.1:8000 =="
python -m heliostream serve --demo
