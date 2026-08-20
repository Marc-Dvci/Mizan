"""Spatial fields: a stationary Gaussian random field for the truth, pilot points
for the estimator.

The two use different parameterisations on purpose. The truth conductivity is a
dense exponential-covariance field on the 1 km grid; the estimator may only move a
5 x 5 array of pilot points interpolated onto the 2 km grid. The inversion therefore
never recovers its own basis, which is the standard guard against the inverse crime.
"""
from __future__ import annotations

import numpy as np

from . import config as C


def grf(grid: C.Grid, corr_len_m: float, sigma: float, seed: int) -> np.ndarray:
    """Zero-mean Gaussian random field with an exponential covariance, shape (n, n)."""
    n = grid.ncol
    rng = np.random.default_rng(seed)
    m = 2 * n
    fx = np.fft.fftfreq(m, d=grid.delr_m)
    kx, ky = np.meshgrid(fx, fx)
    k = np.sqrt(kx ** 2 + ky ** 2)
    # spectral density of a 2-D exponential covariance
    a = corr_len_m / 3.0
    s = (2.0 * np.pi * a ** 2) / (1.0 + (2.0 * np.pi * a * k) ** 2) ** 1.5
    noise = rng.standard_normal((m, m))
    fld = np.real(np.fft.ifft2(np.fft.fft2(noise) * np.sqrt(s)))[:n, :n]
    return sigma * (fld - fld.mean()) / fld.std()


def pilot_locations(grid: C.Grid, npp: int = 5) -> np.ndarray:
    """Pilot-point coordinates in metres, shape (npp*npp, 2)."""
    c = (np.arange(npp) + 0.5) * C.DOMAIN_KM * 1000.0 / npp
    xx, yy = np.meshgrid(c, c)
    return np.column_stack([xx.ravel(), yy.ravel()])


def pilot_to_grid(grid: C.Grid, pts: np.ndarray, vals: np.ndarray,
                  power: float = 2.0) -> np.ndarray:
    """Inverse-distance interpolation of pilot-point values onto `grid`."""
    x, y = grid.centers()
    d2 = (x[..., None] - pts[:, 0]) ** 2 + (y[..., None] - pts[:, 1]) ** 2
    w = 1.0 / np.maximum(d2, 1.0) ** (power / 2.0)
    return (w * vals).sum(axis=-1) / w.sum(axis=-1)


def upscale_mask(fine: np.ndarray, factor: int) -> np.ndarray:
    """Any-overlap upscaling of a boolean mask, as a pivot delineation would rasterise."""
    n = fine.shape[0] // factor
    return fine[: n * factor, : n * factor].reshape(n, factor, n, factor).any(axis=(1, 3))
