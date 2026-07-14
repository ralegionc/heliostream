"""Global configuration: paths, feature schema, horizons, physics constants."""
from __future__ import annotations

from pathlib import Path

# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "artifacts" / "data"
MODEL_DIR = ROOT / "artifacts" / "models"
REPORT_DIR = ROOT / "artifacts" / "reports"
for _d in (DATA_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Modelling schema ----------------------------------------------------
# Raw solar-wind channels we ingest / simulate (hourly cadence).
RAW_COLS = ["bt", "bz", "by", "v", "n"]  # nT, nT, nT, km/s, cm^-3

# Engineered features fed to the network (see data/features.py).
FEATURE_COLS = ["bt", "bz", "by", "v", "n", "bs", "vbs", "pdyn", "sin_clock_half4"]

TARGET_COL = "dst"  # nT

LOOKBACK = 24          # hours of history the encoder sees
HORIZONS = [1, 2, 3, 4, 5, 6]   # forecast lead times (hours ahead)
MAX_HORIZON = max(HORIZONS)

STORM_THRESHOLD = -50.0   # Dst (nT) below this counts as "storm time"
INTENSE_THRESHOLD = -100.0

# --- Physics constants (O'Brien & McPherron 2000) ------------------------
# dDst*/dt = Q - Dst*/tau ; Dst = Dst* + B*sqrt(Pdyn) - C
OBM_EC = 0.49      # mV/m   coupling threshold
OBM_A = 4.4        # nT/hr per mV/m   injection slope
OBM_TAU0 = 2.40    # hr
OBM_TAU_B = 9.74
OBM_TAU_C = 4.69
PRESS_B = 7.26     # nT / nPa^0.5
PRESS_C = 11.0     # nT
PROTON_MASS_FACTOR = 1.6726e-6  # Pdyn[nPa] = k * n[cm^-3] * V[km/s]^2

SEED = 1337
