"""The four observation operators, and the state extraction they run on.

Each operator maps a simulated aquifer state to the quantity an instrument actually
reports, including the parts of the instrument that are themselves unknown: GRACE
leakage and the external mass signal, the non-hydraulic InSAR ramp, the consumptive
fraction behind an evapotranspiration retrieval, and the pre-canopy irrigation that
no evapotranspiration product can see.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import flopy

from . import config as C
from . import forcing as F


# --------------------------------------------------------------------------- states
@dataclass
class State:
    head0: np.ndarray         # (NLAY, nrow, ncol) pre-development steady state
    head: np.ndarray          # (NPER, NLAY, nrow, ncol)
    comp: np.ndarray          # (NPER, NLAY, nrow, ncol) total compaction, m
    area_m2: float
    sy: float

    @property
    def subsidence(self) -> np.ndarray:
        """Land-surface subsidence, m, positive down, shape (NPER, nrow, ncol)."""
        return self.comp.sum(axis=1)

    @property
    def storage_depletion(self) -> np.ndarray:
        """Cumulative water released from basin storage, m3, shape (NPER,).

        Two terms. Skeletal and coarse-grained release equals compaction in the
        head-based CSUB formulation, so its domain integral is one term directly.
        The water table in the unconfined upper unit contributes the second, through
        specific yield. Their sum is what a gravity mission measures over the basin.
        """
        skeletal = self.comp.sum(axis=(1, 2, 3))
        watertable = self.sy * (self.head0[0][None] - self.head[:, 0]).sum(axis=(1, 2))
        return (skeletal + watertable) * self.area_m2


def read_inelastic(ws: Path, grid: C.Grid) -> np.ndarray:
    """Inelastic compaction field written by CSUB, m, shape (NPER, NLAY, nrow, ncol).

    Multiplied by cell area and summed, this is the storage capacity that inelastic
    compaction has destroyed. It does not come back when heads recover.
    """
    f = flopy.utils.HeadFile(str(Path(ws) / "mizan.cmpi"),
                             text="CSUB-INELASTIC")
    d = np.asarray(f.get_alldata())
    f.close()
    return d[1:]


def read_state(ws: Path, grid: C.Grid, sy: float = C.SY) -> State:
    hds = flopy.utils.HeadFile(str(Path(ws) / "mizan.hds"))
    h = np.asarray(hds.get_alldata())
    hds.close()
    cmp = flopy.utils.HeadFile(str(Path(ws) / "mizan.cmp"), text="CSUB-COMPACTION")
    c = np.asarray(cmp.get_alldata())
    cmp.close()
    return State(head0=h[0], head=h[1:], comp=c[1:], area_m2=grid.delr_m ** 2, sy=sy)


# --------------------------------------------------------------------------- geometry
@dataclass
class Geometry:
    well_xy: np.ndarray        # (nw, 2) metres
    well_seen: np.ndarray      # (nw, NPER) bool, months with a measurement
    insar_xy: np.ndarray       # (npx, 2) metres
    insar_ref_xy: np.ndarray   # (2,)
    insar_epochs: np.ndarray   # (nt,) stress-period indices carrying an acquisition


def make_geometry(mask_truth: np.ndarray, grid_truth: C.Grid, seed: int = 7) -> Geometry:
    """Instrument locations, defined in physical coordinates so both grids share them.

    Wells cluster on irrigated land, as real monitoring networks do. InSAR pixels are
    on a regular 6 km lattice, with the reference far from any pumping.
    """
    rng = np.random.default_rng(seed)
    x, y = grid_truth.centers()
    nw = int(C.HEAD_CELL_FRACTION * mask_truth.sum())
    pool_irr = np.argwhere(mask_truth)
    pool_all = np.argwhere(np.ones_like(mask_truth, dtype=bool))
    take_irr = pool_irr[rng.choice(len(pool_irr), size=int(0.7 * nw), replace=False)]
    take_any = pool_all[rng.choice(len(pool_all), size=nw - len(take_irr), replace=False)]
    cells = np.vstack([take_irr, take_any])
    well_xy = np.column_stack([x[cells[:, 0], cells[:, 1]], y[cells[:, 0], cells[:, 1]]])
    seen = rng.random((len(well_xy), C.NPER)) < 0.35

    step = int(round(C.INSAR_PIXEL_KM * 1000.0 / grid_truth.delr_m))
    ij = np.array([[i, j] for i in range(step // 2, grid_truth.nrow, step)
                   for j in range(step // 2, grid_truth.ncol, step)])
    insar_xy = np.column_stack([x[ij[:, 0], ij[:, 1]], y[ij[:, 0], ij[:, 1]]])
    epochs = np.arange(C.INSAR_STACK_MONTHS - 1, C.NPER, C.INSAR_STACK_MONTHS)
    return Geometry(well_xy=well_xy, well_seen=seen, insar_xy=insar_xy,
                    insar_ref_xy=np.array([2000.0, 98000.0]), insar_epochs=epochs)


def cells_of(grid: C.Grid, xy: np.ndarray) -> np.ndarray:
    """Map physical coordinates to (row, col) on `grid`."""
    idx = np.clip((xy / grid.delr_m).astype(int), 0, grid.ncol - 1)
    return np.column_stack([idx[:, 1], idx[:, 0]])


# --------------------------------------------------------------------------- operators
def op_head(state: State, geom: Geometry, grid: C.Grid) -> np.ndarray:
    """Simulated head anomalies at monitoring wells, m, flattened over the seen mask.

    Each well is referred to the mean of its own record. A regional model cannot be
    held to the absolute head at a well whose casing elevation and screened interval
    carry their own error, and the quantity that carries the abstraction signal is the
    change, not the datum.
    """
    rc = cells_of(grid, geom.well_xy)
    h = state.head[:, C.CSUB_LAYER, rc[:, 0], rc[:, 1]].T        # (nw, NPER)
    a = np.where(geom.well_seen, h, np.nan)
    return (h - np.nanmean(a, axis=1, keepdims=True))[geom.well_seen]


def op_insar(state: State, geom: Geometry, grid: C.Grid,
             ramp: np.ndarray) -> np.ndarray:
    """Simulated line-of-sight displacement, m, referenced to a stable pixel.

    `ramp` is the non-hydraulic component (constant, east and north gradient) in
    m/yr, which the estimator carries as a nuisance parameter.
    """
    rc = cells_of(grid, geom.insar_xy)
    rr = cells_of(grid, geom.insar_ref_xy[None, :])[0]
    s = state.subsidence[:, rc[:, 0], rc[:, 1]]                  # (NPER, npx)
    sref = state.subsidence[:, rr[0], rr[1]][:, None]
    inc = np.deg2rad(C.INSAR_INCIDENCE_DEG)
    los = -np.cos(inc) * (s - sref)
    t_yr = (np.arange(C.NPER) + 1) / 12.0
    xk = (geom.insar_xy[:, 0] - geom.insar_ref_xy[0]) / 1000.0
    yk = (geom.insar_xy[:, 1] - geom.insar_ref_xy[1]) / 1000.0
    los = los + t_yr[:, None] * (ramp[0] + ramp[1] * xk[None, :] / 100.0
                                 + ramp[2] * yk[None, :] / 100.0)
    los = los[geom.insar_epochs]
    return (los - los.mean(axis=0, keepdims=True)).ravel()


def op_grace(state: State, alpha: float, drift: np.ndarray) -> np.ndarray:
    """Simulated basin groundwater storage anomaly, mm equivalent water height.

    `alpha` is the mascon leakage factor and `drift` the external mass signal that
    the GLDAS removal does not account for; both are estimated, not assumed.
    """
    area = C.DOMAIN_KM ** 2 * 1.0e6
    gws_mm = -state.storage_depletion / area * 1000.0
    t_yr = (np.arange(C.NPER) + 1) / 12.0
    ext = drift[0] * t_yr + drift[1] * np.sin(2 * np.pi * t_yr) + drift[2] * np.cos(2 * np.pi * t_yr)
    return alpha * gws_mm + ext


def op_et(q_annual: np.ndarray, eta: np.ndarray, preplant: np.ndarray) -> np.ndarray:
    """Simulated irrigation evapotranspiration, m3, shape (NDIST * NPER,).

    Only water applied under an established canopy is retrieved as irrigation ET.
    The pre-canopy share `preplant` leaves the aquifer and never appears here, which
    is the reason an evapotranspiration-only account is a lower bound.
    """
    shape = C.MONTHLY_SHAPE.copy()
    for m in C.PREPLANT_MONTHS:
        shape[m] = 0.0
    shape = shape / shape.sum()
    canopy = np.tile(shape, C.NYEAR)                                   # (NPER,)
    annual_canopy = q_annual * (1.0 - preplant)[:, None]               # (NDIST, NYEAR)
    monthly = np.repeat(annual_canopy, 12, axis=1) * canopy[None, :]
    return (eta[:, None] * monthly).ravel()


# --------------------------------------------------------------------------- noise
def op_meter(q_annual: np.ndarray) -> np.ndarray:
    """A metered district: annual abstraction read directly, m3, shape (NDIST*NYEAR,).

    A meter does not observe the aquifer. It observes the decision variable, which is
    why a small number of them changes the problem out of proportion to their cost.
    """
    return q_annual.ravel()


METER_REL_SIGMA = 0.08


def add_noise(truth: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Corrupt the truth observation vectors with each instrument's error model."""
    out = {}
    out["et"] = truth["et"] * (1.0 + C.ET_REL_SIGMA * rng.standard_normal(truth["et"].shape))
    out["grace"] = truth["grace"] + C.GRACE_SIGMA_MM * rng.standard_normal(truth["grace"].shape)
    n = truth["insar"].size
    out["insar"] = truth["insar"] + C.INSAR_SIGMA_M * rng.standard_normal(n)
    out["head"] = truth["head"] + C.HEAD_SIGMA_M * rng.standard_normal(truth["head"].shape)
    if "meter" in truth:
        out["meter"] = truth["meter"] * (
            1.0 + METER_REL_SIGMA * rng.standard_normal(truth["meter"].shape))
    return out


def obs_sigma(truth: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Assumed observation standard deviations, one vector per leg."""
    return {
        "et": np.maximum(C.ET_REL_SIGMA * np.abs(truth["et"]), 1.0e4),
        "grace": np.full(truth["grace"].shape, C.GRACE_SIGMA_MM),
        "insar": np.full(truth["insar"].shape, C.INSAR_SIGMA_M),
        "head": np.full(truth["head"].shape, C.HEAD_SIGMA_M),
        **({"meter": np.maximum(METER_REL_SIGMA * np.abs(truth["meter"]), 1.0e5)}
           if "meter" in truth else {}),
    }
