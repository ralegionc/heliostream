"""Ring-current physics: coupling functions and the O'Brien-McPherron (2000)
empirical Dst model. Used both as a standalone baseline and as the structural
prior baked into the hybrid neural model.

Reference: O'Brien, T. P., & McPherron, R. L. (2000), JGR 105(A4).
    dDst*/dt = Q(t) - Dst*/tau
    Q   = -a (VBs - Ec)   for VBs > Ec, else 0        [nT/hr]
    tau = tau0 * exp(b / (c + VBs))                    [hr]
    Dst = Dst* + B*sqrt(Pdyn) - C                      [nT]
"""
from __future__ import annotations

import numpy as np

from .. import config as C


def dynamic_pressure(n: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Solar-wind dynamic pressure [nPa] from density [cm^-3] and speed [km/s]."""
    return C.PROTON_MASS_FACTOR * n * v ** 2


def coupling_vbs(v: np.ndarray, bz: np.ndarray) -> np.ndarray:
    """Rectified solar-wind electric field VBs [mV/m].

    Bs = max(0, -Bz). E = V*Bs with unit factor 1e-3 (km/s * nT -> mV/m).
    """
    bs = np.clip(-bz, 0.0, None)
    return v * bs * 1e-3


def obm_injection(vbs: np.ndarray) -> np.ndarray:
    """Injection term Q [nT/hr]."""
    q = -C.OBM_A * (vbs - C.OBM_EC)
    return np.where(vbs > C.OBM_EC, q, 0.0)


def obm_tau(vbs: np.ndarray) -> np.ndarray:
    """Decay timescale tau [hr]."""
    return C.OBM_TAU0 * np.exp(C.OBM_TAU_B / (C.OBM_TAU_C + vbs))


def pressure_correction(dst_star: np.ndarray, pdyn: np.ndarray) -> np.ndarray:
    """Convert pressure-corrected Dst* to observed Dst."""
    return dst_star + C.PRESS_B * np.sqrt(np.clip(pdyn, 0, None)) - C.PRESS_C


def remove_pressure(dst: np.ndarray, pdyn: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pressure_correction` (observed Dst -> Dst*)."""
    return dst - C.PRESS_B * np.sqrt(np.clip(pdyn, 0, None)) + C.PRESS_C


def integrate_obm(
    v: np.ndarray,
    n: np.ndarray,
    bz: np.ndarray,
    dst0: float,
    dt: float = 1.0,
) -> np.ndarray:
    """Forward-integrate the O'Brien-McPherron model over a solar-wind sequence.

    Parameters
    ----------
    v, n, bz : arrays of length T (hourly solar wind).
    dst0 : initial observed Dst [nT].
    dt : timestep [hr].

    Returns array of observed Dst of length T (the model's own trajectory).
    """
    v = np.asarray(v, float)
    n = np.asarray(n, float)
    bz = np.asarray(bz, float)
    T = len(v)
    vbs = coupling_vbs(v, bz)
    pdyn = dynamic_pressure(n, v)
    q = obm_injection(vbs)
    tau = obm_tau(vbs)

    dst_star = np.empty(T)
    dst_star[0] = remove_pressure(np.array([dst0]), pdyn[:1])[0]
    for t in range(1, T):
        # RK2 (midpoint) on dDst*/dt = Q - Dst*/tau
        s = dst_star[t - 1]
        k1 = q[t - 1] - s / tau[t - 1]
        s_mid = s + 0.5 * dt * k1
        qm = 0.5 * (q[t - 1] + q[t])
        taum = 0.5 * (tau[t - 1] + tau[t])
        k2 = qm - s_mid / taum
        dst_star[t] = s + dt * k2
    return pressure_correction(dst_star, pdyn)
