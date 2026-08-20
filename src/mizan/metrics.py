"""Scoring for the L0 experiment.

The scored quantity is district-annual abstraction, which is the decision variable a
regulator can act on, not head and not storage. Interval calibration is scored
alongside the point error, because an abstraction estimate without a defensible error
bar cannot enter a chance constraint.
"""
from __future__ import annotations

import numpy as np

from . import config as C

MCM = 1.0e6


def et_annual(obs_et: np.ndarray) -> np.ndarray:
    """District-annual irrigation evapotranspiration, m3, shape (NDIST, NYEAR)."""
    return obs_et.reshape(C.NDIST, C.NYEAR, 12).sum(axis=2)


def baseline_open_loop(obs_et: np.ndarray, efficiency: float | None = None) -> np.ndarray:
    """The published open-loop account: consumptive use divided by a fixed efficiency."""
    e = C.BASELINE_FIXED_EFFICIENCY if efficiency is None else efficiency
    return et_annual(obs_et) / e


def oracle_efficiency(obs_et: np.ndarray, q_true: np.ndarray) -> float:
    """The single global efficiency that minimises open-loop absolute error.

    Not available to a practitioner: it is fitted against the answer. It exists so
    that the comparison is against the best that the open-loop form can do, rather
    than against its published constant.
    """
    e = et_annual(obs_et)
    grid = np.linspace(0.40, 0.99, 600)
    err = np.array([np.abs(e / g - q_true).mean() for g in grid])
    return float(grid[int(err.argmin())])


def point_scores(q_hat: np.ndarray, q_true: np.ndarray) -> dict[str, float]:
    """Error of a district-annual abstraction estimate, in Mm3/yr and per cent."""
    d = q_hat - q_true
    return {
        "mae_mcm": float(np.abs(d).mean() / MCM),
        "bias_mcm": float(d.mean() / MCM),
        "rmse_mcm": float(np.sqrt((d ** 2).mean()) / MCM),
        "mape_pct": float((np.abs(d) / q_true).mean() * 100.0),
        "basin_bias_pct": float((q_hat.sum() - q_true.sum()) / q_true.sum() * 100.0),
    }


def coverage(ens: np.ndarray, q_true: np.ndarray, levels=(0.50, 0.80, 0.90)) -> dict[str, float]:
    """Empirical coverage of posterior intervals.

    `ens` has shape (ne, NDIST, NYEAR).
    """
    out = {}
    for lv in levels:
        lo = np.quantile(ens, 0.5 - lv / 2.0, axis=0)
        hi = np.quantile(ens, 0.5 + lv / 2.0, axis=0)
        out[f"cover_{int(lv*100)}"] = float(((q_true >= lo) & (q_true <= hi)).mean())
    lo = np.quantile(ens, 0.05, axis=0)
    hi = np.quantile(ens, 0.95, axis=0)
    out["width90_mcm"] = float((hi - lo).mean() / MCM)
    return out


def crps(ens: np.ndarray, q_true: np.ndarray) -> float:
    """Continuous ranked probability score, Mm3/yr, averaged over district-years."""
    e = np.sort(ens, axis=0) / MCM
    t = q_true / MCM
    ne = e.shape[0]
    spread = (((2 * np.arange(1, ne + 1) - ne - 1)[:, None, None] * e).sum(axis=0)
              * 2.0 / (ne * ne))
    return float((np.abs(e - t[None]).mean(axis=0) - 0.5 * spread).mean())


def resolved_directions(ens: np.ndarray, prior_ens: np.ndarray) -> dict:
    """How much of the district-annual abstraction vector the data actually resolved.

    Both ensembles are taken in log abstraction. The generalised eigenvalues of the
    posterior against the prior covariance give, for every direction of the 180
    district-year vector, the fraction of prior variance that survived. Directions
    with a ratio near one are the ones the observation set could not see, and are
    reported rather than smoothed over.
    """
    A = np.log10(np.maximum(ens.reshape(ens.shape[0], -1), 1.0))
    B = np.log10(np.maximum(prior_ens.reshape(prior_ens.shape[0], -1), 1.0))
    n = A.shape[1]
    Cpost = np.cov(A.T) + 1e-10 * np.eye(n)
    Cpri = np.cov(B.T) + 1e-10 * np.eye(n)
    L = np.linalg.cholesky(Cpri)
    Li = np.linalg.inv(L)
    M = Li @ Cpost @ Li.T
    w, v = np.linalg.eigh((M + M.T) / 2.0)
    order = np.argsort(w)
    ratio = np.clip(w[order], 0.0, None)
    return {
        "variance_ratio": ratio,
        "directions": (L.T @ v[:, order]),
        "n_resolved_90": int((ratio < 0.10).sum()),
        "n_resolved_50": int((ratio < 0.50).sum()),
        "n_unresolved": int((ratio > 0.90).sum()),
        "effective_dim": float((1.0 - ratio).sum()),
    }
