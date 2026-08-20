"""Domain, time and physical constants for the Mizan L0 synthetic twin.

The synthetic basin is a stylised fossil-aquifer irrigation district: a confined
sandstone aquifer under a compressible interbedded upper unit, effectively no
recharge, and centre-pivot abstraction concentrated in nine management districts.

Two grids are defined. `TRUTH` is the grid the synthetic reality is generated on.
`EST` is the coarser grid the estimator is allowed to use. They differ so that the
inversion never recovers its own discretisation.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# ----------------------------------------------------------------------------- domain
DOMAIN_KM = 100.0            # square basin, 100 km on a side
NLAY = 2
TOP = 20.0
BOTM = [-120.0, -500.0]      # unconfined upper unit, confined main aquifer

# ----------------------------------------------------------------------------- time
NYEAR = 20
NPER = NYEAR * 12            # monthly stress periods, preceded by one steady state
START_YEAR = 2005

# ----------------------------------------------------------------------------- districts
NDIST = 9                    # 3 x 3 management districts

# ----------------------------------------------------------------------------- hydraulics
K1 = 0.5                     # m/d, upper unit
K2 = 6.0                     # m/d, main aquifer
SY = 0.08                    # specific yield of the unconfined upper unit
SS_CG = 1.0e-6               # 1/m, coarse-grained specific storage
RECHARGE_MM_YR = 0.6         # effectively zero: fossil aquifer
GHB_COND = 5.0e-3            # m2/d per cell, weak lateral connection on two edges
H_INIT = 0.0                 # m, initial head (datum = pre-development potentiometric)

# CSUB, head-based formulation: preconsolidation is a head, which is the quantity
# the allocation layer constrains directly.
CSUB_LAYER = 1               # interbeds sit inside the pumped confined aquifer
CSUB_THICK_FRAC = 0.10       # interbed fraction of that layer
CSUB_SSV = 3.5e-4            # 1/m, inelastic (virgin) skeletal specific storage
CSUB_SSE = 6.0e-6            # 1/m, elastic skeletal specific storage
CSUB_THETA = 0.3
PCS_OFFSET = 12.0            # m, initial preconsolidation head sits this far below h0

# ----------------------------------------------------------------------------- abstraction
# Annual district abstraction, million m3/yr, at the start of the record.
DIST_Q0_MCM = np.array([180., 95., 240., 60., 310., 140., 85., 200., 55.])
# Consumptive (beneficial ET) fraction of pumped water, per district. The published
# open-loop method assumes a single fixed 0.80 for every district and every year.
DIST_ETA = np.array([0.82, 0.74, 0.88, 0.69, 0.85, 0.79, 0.72, 0.86, 0.76])
# Fraction of the non-consumed water that returns to the aquifer as deep percolation.
RETURN_FRAC = 0.30
# Months in which irrigation is applied before the canopy exists. Satellite ET
# attributes this to bare-soil evaporation, so it is invisible to the ET leg.
PREPLANT_MONTHS = (10, 11)
# Share of annual abstraction applied pre-canopy, per district.
DIST_PREPLANT = np.array([0.09, 0.14, 0.07, 0.16, 0.10, 0.12, 0.15, 0.06, 0.13])

MONTHLY_SHAPE = np.array(
    [0.030, 0.035, 0.060, 0.095, 0.130, 0.150, 0.155, 0.140, 0.100, 0.055, 0.030, 0.020]
)

# ----------------------------------------------------------------------------- observation error
ET_REL_SIGMA = 0.12          # OpenET / WaPOR monthly field-scale relative error
GRACE_SIGMA_MM = 14.0        # mm equivalent water height, mascon-scale
GRACE_FOOTPRINT_KM = 300.0
INSAR_SIGMA_M = 0.006        # m, LOS after time-series inversion
INSAR_INCIDENCE_DEG = 39.0
# Sentinel-1 revisits every 12 days at 20 m posting. The stack is decimated to the
# correlation scale of the tropospheric noise so that the observations entering the
# likelihood are close to independent, rather than inflating an error covariance.
INSAR_PIXEL_KM = 12.0
INSAR_STACK_MONTHS = 3
HEAD_SIGMA_M = 0.5
HEAD_CELL_FRACTION = 0.05

BASELINE_FIXED_EFFICIENCY = 0.80   # the published open-loop assumption


@dataclass(frozen=True)
class Grid:
    """A structured grid over the fixed physical domain."""

    ncol: int
    delr_m: float

    @property
    def nrow(self) -> int:
        return self.ncol

    @property
    def ncell(self) -> int:
        return NLAY * self.ncol * self.ncol

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centre coordinates in metres, as (x, y) meshgrids of shape (nrow, ncol)."""
        c = (np.arange(self.ncol) + 0.5) * self.delr_m
        return np.meshgrid(c, c)

    def district_map(self) -> np.ndarray:
        """District index 0..NDIST-1 for every cell, shape (nrow, ncol)."""
        third = self.ncol / 3.0
        i = np.minimum((np.arange(self.nrow) / third).astype(int), 2)
        j = np.minimum((np.arange(self.ncol) / third).astype(int), 2)
        return (i[:, None] * 3 + j[None, :]).astype(int)


TRUTH = Grid(ncol=100, delr_m=1000.0)
EST = Grid(ncol=50, delr_m=2000.0)
