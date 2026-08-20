"""Irrigated-field geometry and the abstraction forcing that drives the aquifer."""
from __future__ import annotations

import numpy as np

from . import config as C


def pivot_mask(grid: C.Grid, seed: int = 11) -> np.ndarray:
    """Boolean (nrow, ncol) mask of irrigated cells, clustered inside each district.

    Cluster centres and radii are a property of the landscape, so they are generated
    from physical coordinates and are identical on any grid.
    """
    rng = np.random.default_rng(seed)
    x, y = grid.centers()
    mask = np.zeros((grid.nrow, grid.ncol), dtype=bool)
    third = C.DOMAIN_KM * 1000.0 / 3.0
    for d in range(C.NDIST):
        i, j = divmod(d, 3)
        cx0, cy0 = (j + 0.5) * third, (i + 0.5) * third
        nclust = 4
        for _ in range(nclust):
            cx = cx0 + rng.uniform(-0.30, 0.30) * third
            cy = cy0 + rng.uniform(-0.30, 0.30) * third
            r = rng.uniform(0.10, 0.18) * third
            mask |= ((x - cx) ** 2 + (y - cy) ** 2) < r ** 2
    return mask


def district_annual_abstraction() -> np.ndarray:
    """Truth district-annual abstraction, m3/yr, shape (NDIST, NYEAR).

    Each district rises then falls, mirroring a build-out followed by a restriction
    policy, with district-specific timing so the annual vectors are not collinear.
    """
    rng = np.random.default_rng(23)
    t = np.arange(C.NYEAR)
    out = np.zeros((C.NDIST, C.NYEAR))
    for d in range(C.NDIST):
        peak = rng.uniform(7.0, 13.0)
        width = rng.uniform(6.0, 11.0)
        shape = np.exp(-0.5 * ((t - peak) / width) ** 2)
        shape = 0.55 + 0.75 * shape / shape.max()
        wiggle = 1.0 + 0.09 * rng.standard_normal(C.NYEAR)
        out[d] = C.DIST_Q0_MCM[d] * 1.0e6 * shape * wiggle
    return np.maximum(out, 0.0)


def monthly_abstraction(annual: np.ndarray, preplant: np.ndarray) -> np.ndarray:
    """Split district-annual volumes into monthly rates, m3/d, shape (NDIST, 12*nyear).

    The pre-canopy share is moved into `PREPLANT_MONTHS`; the remainder follows the
    crop-demand shape. This is the component the ET leg cannot see.
    """
    base = C.MONTHLY_SHAPE.copy()
    for m in C.PREPLANT_MONTHS:
        base[m] = 0.0
    base = base / base.sum()

    shape = np.zeros((C.NDIST, 12))
    shape[:] = base[None, :] * (1.0 - preplant)[:, None]
    for m in C.PREPLANT_MONTHS:
        shape[:, m] = preplant / len(C.PREPLANT_MONTHS)

    days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=float)
    nyear = annual.shape[1]
    out = np.zeros((C.NDIST, nyear * 12))
    for y in range(nyear):
        for m in range(12):
            out[:, y * 12 + m] = annual[:, y] * shape[:, m] / days[m]
    return out


def canopy_share() -> np.ndarray:
    """Monthly distribution of canopy-period irrigation, length 12, sums to one.

    Recoverable from an NDVI time series, so the estimator is allowed to know it.
    """
    shape = C.MONTHLY_SHAPE.copy()
    for m in C.PREPLANT_MONTHS:
        shape[m] = 0.0
    return shape / shape.sum()


def well_rates(grid: C.Grid, monthly: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-cell pumping rate, m3/d, shape (NPER, nrow, ncol), spread over pivot cells.

    Within a district the rate is distributed over irrigated cells with a fixed
    spatial weight, so district totals are preserved exactly on either grid.
    """
    dmap = grid.district_map()
    w = np.zeros((grid.nrow, grid.ncol))
    for d in range(C.NDIST):
        sel = mask & (dmap == d)
        n = int(sel.sum())
        if n:
            w[sel] = 1.0 / n
    rates = np.zeros((C.NPER, grid.nrow, grid.ncol))
    for d in range(C.NDIST):
        sel = mask & (dmap == d)
        if sel.any():
            rates[:, sel] = monthly[d][:, None] * w[sel][None, :]
    return rates


def return_flow(grid: C.Grid, monthly: np.ndarray, mask: np.ndarray,
                eta: np.ndarray) -> np.ndarray:
    """Deep percolation returning to the aquifer, m3/d, shape (NPER, nrow, ncol)."""
    frac = C.RETURN_FRAC * (1.0 - eta)
    return well_rates(grid, monthly * frac[:, None], mask)
