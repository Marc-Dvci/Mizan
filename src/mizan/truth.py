"""The hidden reality of the L0 experiment.

Every value here is withheld from the estimator. Each one is offset from the prior
mean the estimator is given, so that recovering it requires information from the
data rather than from the prior.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from . import fields, forcing
from .model import Params

# Values the estimator never sees. The second column of the comment is the prior
# mean the estimator starts from, defined in `prior.py`.
SS_CG = 1.6e-6          # prior mean 1.0e-6
SY = 0.080              # prior mean 0.110
CSUB_SSV = 3.5e-4       # prior mean 2.2e-4
CSUB_SSE = 6.0e-6       # prior mean 1.0e-5
THICK_FRAC = 0.10       # prior mean 0.075
PCS_OFFSET = 12.0       # prior mean 18.0
RECHARGE = 0.6          # prior mean 1.0
GHB_COND = 5.0e-3       # prior mean 3.0e-3

GRACE_ALPHA = 0.78      # prior mean 0.85, so the shipped prior is wrong by 1.75 sd
GRACE_DRIFT = np.array([-0.8, 3.5, -2.2])      # mm/yr, mm, mm
INSAR_RAMP = np.array([0.0022, 0.0016, -0.0011])   # m/yr, m/yr per 100 km, per 100 km


def logk_fields(grid: C.Grid) -> tuple[np.ndarray, np.ndarray]:
    return (np.log10(C.K1) + fields.grf(grid, 15000.0, 0.35, seed=101),
            np.log10(C.K2) + fields.grf(grid, 25000.0, 0.30, seed=102))


def params(grid: C.Grid) -> Params:
    k1, k2 = logk_fields(grid)
    ann = forcing.district_annual_abstraction()
    mon = forcing.monthly_abstraction(ann, C.DIST_PREPLANT)
    return Params(logk1=k1, logk2=k2, ss_cg=SS_CG, sy=SY, csub_ssv=CSUB_SSV, csub_sse=CSUB_SSE,
                  thick_frac=THICK_FRAC, pcs_offset=PCS_OFFSET, recharge_mm_yr=RECHARGE,
                  ghb_cond=GHB_COND, q_monthly=mon, eta=C.DIST_ETA)
