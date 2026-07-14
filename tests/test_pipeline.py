"""Fast tests exercising the offline (synthetic) path end to end."""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

from heliostream.data import synthetic, features as F
from heliostream.models import physics as phys
from heliostream.models.hybrid import PhysicsInformedDst
from heliostream import train as T, evaluate as E, serve


def test_synthetic_is_physical():
    df = synthetic.simulate(24 * 120, seed=1)
    assert {"bt", "bz", "by", "v", "n", "dst"} <= set(df.columns)
    assert (df.dst < 40).all()                      # Dst never strongly positive
    assert (df.dst < -50).sum() > 0                 # contains storms
    assert (np.abs(df.bz) <= df.bt + 1e-6).all()    # |Bz| <= Bt


def test_obm_recovers_quiet_baseline():
    # zero southward field -> no injection -> Dst decays toward pressure baseline
    T_ = 48
    v = np.full(T_, 400.0); n = np.full(T_, 5.0); bz = np.full(T_, 2.0)
    traj = phys.integrate_obm(v, n, bz, dst0=-80.0)
    assert traj[-1] > traj[0]                        # recovery (rises toward 0)


def test_windowing_shapes():
    df = F.engineer(synthetic.simulate(24 * 60, seed=2))
    norm = F.fit_normalizer(df)
    X, y, dst0, swf, idx = F.make_windows(df, norm)
    assert X.shape[1:] == (F.C.LOOKBACK, len(F.C.FEATURE_COLS))
    assert y.shape[1] == len(F.C.HORIZONS)
    assert len(X) == len(y) == len(dst0) == len(swf)


def test_hybrid_forward_and_grad():
    m = PhysicsInformedDst(hidden=16)
    x = torch.randn(8, F.C.LOOKBACK, len(F.C.FEATURE_COLS))
    dst0 = torch.full((8,), -20.0)
    swf = torch.tensor([[450., 6., -5.]] * 8)
    mean, logvar = m(x, dst0, swf)
    assert mean.shape == (8, len(F.C.HORIZONS))
    mean.sum().backward()                            # differentiable through ODE
    assert next(m.parameters()).grad is not None


def test_training_improves_and_calibrates():
    df = synthetic.simulate(24 * 365 * 2, seed=3)
    model, norm, hist = T.train_model(df, "hybrid", epochs=8, verbose=False)
    assert hist[-1]["val_rmse"] < hist[0]["val_rmse"]     # learning happened
    dfe = F.engineer(df); _, va, te = F.time_split(dfe)
    rep, (mean, std, y, factors) = E.full_report(model, norm, va, te)
    assert rep["model"]["rmse_all"] < rep["persistence"]["rmse_all"]  # beats persistence
    assert 0.82 <= rep["conformal"]["coverage_conformal"] <= 0.98      # ~calibrated


def test_nowcast_schema():
    df = synthetic.simulate(24 * 365, seed=4)
    model, norm, _ = T.train_model(df, "hybrid", epochs=3, verbose=False)
    window = df.iloc[-F.C.LOOKBACK:][F.C.RAW_COLS]
    nc = serve.build_nowcast(model, norm, window, dst0=-25.0, source="test")
    assert len(nc["forecast"]) == len(F.C.HORIZONS)
    for f in nc["forecast"]:
        assert f["lo"] <= f["mean"] <= f["hi"]
    assert "label" in nc["alert"]
