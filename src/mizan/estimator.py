"""The estimator's view of the problem: what it may vary, what it believes a priori,
and how a parameter vector becomes a simulated observation vector.

The estimator works on the coarse grid, parameterises conductivity with pilot points
rather than the dense field the truth was drawn from, and carries the instrument
nuisances (gravity leakage and external mass, the non-hydraulic interferometric ramp,
the consumptive fraction, the pre-canopy share) as unknowns rather than assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config as C
from . import fields, forcing, model as M, observations as O

NPP = 5                                  # pilot points per side, per layer
NPILOT = NPP * NPP

# --------------------------------------------------------------------------- layout
def _layout() -> dict[str, slice]:
    idx, out = 0, {}
    for name, n in [
        ("logq", C.NDIST * C.NYEAR),
        ("eta", C.NDIST),
        ("preplant", C.NDIST),
        ("logk1", NPILOT),
        ("logk2", NPILOT),
        ("log_ss", 1), ("log_sy", 1), ("log_ssv", 1), ("log_sse", 1),
        ("log_thick", 1), ("log_pcs", 1), ("log_rch", 1), ("log_ghb", 1),
        ("grace_alpha", 1), ("grace_drift", 3), ("insar_ramp", 3),
    ]:
        out[name] = slice(idx, idx + n)
        idx += n
    out["_n"] = idx
    return out


LAYOUT = _layout()
NPAR = LAYOUT["_n"]


@dataclass
class Prior:
    mean: np.ndarray
    sd: np.ndarray
    names: list[str]
    lo: np.ndarray = None
    hi: np.ndarray = None


def prior(mask_est: np.ndarray, grid: C.Grid) -> Prior:
    """Prior mean and standard deviation for every parameter, in native units.

    The abstraction prior is deliberately independent of the evapotranspiration data,
    so that no leg enters twice: it is a uniform 1.0 m/yr applied depth over the
    delineated irrigated area of each district, with a factor-of-two spread.
    """
    mean = np.zeros(NPAR)
    sd = np.zeros(NPAR)
    names: list[str] = []

    dmap = grid.district_map()
    area = np.array([(mask_est & (dmap == d)).sum() * grid.delr_m ** 2 for d in range(C.NDIST)])
    q0 = np.log10(np.maximum(area * 1.0, 1.0e6))                  # 1.0 m/yr applied depth
    mean[LAYOUT["logq"]] = np.repeat(q0, C.NYEAR)
    sd[LAYOUT["logq"]] = 0.30
    names += [f"logq_d{d}_y{y}" for d in range(C.NDIST) for y in range(C.NYEAR)]

    mean[LAYOUT["eta"]] = C.BASELINE_FIXED_EFFICIENCY
    sd[LAYOUT["eta"]] = 0.06
    names += [f"eta_d{d}" for d in range(C.NDIST)]

    mean[LAYOUT["preplant"]] = 0.06
    sd[LAYOUT["preplant"]] = 0.04
    names += [f"preplant_d{d}" for d in range(C.NDIST)]

    mean[LAYOUT["logk1"]] = np.log10(C.K1); sd[LAYOUT["logk1"]] = 0.40
    mean[LAYOUT["logk2"]] = np.log10(C.K2); sd[LAYOUT["logk2"]] = 0.40
    names += [f"logk1_p{i}" for i in range(NPILOT)] + [f"logk2_p{i}" for i in range(NPILOT)]

    for key, m, s in [("log_ss", 1.0e-6, 0.30), ("log_sy", 0.110, 0.15),
                      ("log_ssv", 2.2e-4, 0.30), ("log_sse", 1.0e-5, 0.30),
                      ("log_thick", 0.075, 0.20), ("log_pcs", 18.0, 0.25),
                      ("log_rch", 1.0, 0.30), ("log_ghb", 3.0e-3, 0.40)]:
        mean[LAYOUT[key]] = np.log10(m)
        sd[LAYOUT[key]] = s
        names.append(key)

    # The mascon gain factor is computed from the mascon geometry and a land-surface
    # model, not fitted, so it enters as a tightly constrained known rather than a free
    # nuisance. Left free it absorbs the absolute scale of the abstraction estimate,
    # which is the one thing the gravity leg is there to supply.
    mean[LAYOUT["grace_alpha"]] = 0.85; sd[LAYOUT["grace_alpha"]] = 0.04
    names.append("grace_alpha")
    # In a hyper-arid basin with no surface water and no snow, the trend in total
    # water storage is the trend in groundwater. A freely fitted external trend is
    # degenerate with the mascon gain factor, because basin storage falls close to
    # linearly, and between them they absorb the absolute scale of the answer.
    mean[LAYOUT["grace_drift"]] = 0.0; sd[LAYOUT["grace_drift"]] = np.array([1.0, 4.0, 4.0])
    names += ["grace_drift_trend", "grace_drift_sin", "grace_drift_cos"]
    mean[LAYOUT["insar_ramp"]] = 0.0; sd[LAYOUT["insar_ramp"]] = 0.003
    names += ["insar_ramp_0", "insar_ramp_x", "insar_ramp_y"]

    # Admissible ranges. Every one is a physical statement, not a numerical guard:
    # an efficiency above 0.95 or a specific yield of 0.4 is not a candidate aquifer.
    lo = mean - 4.0 * sd
    hi = mean + 4.0 * sd
    def setb(key, a, b):
        lo[LAYOUT[key]] = a
        hi[LAYOUT[key]] = b
    setb("logq", mean[LAYOUT["logq"]] - 1.2, mean[LAYOUT["logq"]] + 1.2)
    setb("eta", 0.62, 0.95)
    setb("preplant", 0.0, 0.30)
    setb("logk1", np.log10(C.K1) - 1.5, np.log10(C.K1) + 1.5)
    setb("logk2", np.log10(C.K2) - 1.5, np.log10(C.K2) + 1.5)
    setb("log_ss", -7.5, -4.5)
    setb("log_sy", np.log10(0.02), np.log10(0.30))
    setb("log_ssv", np.log10(2e-5), np.log10(4e-3))
    setb("log_sse", np.log10(5e-7), np.log10(2e-4))
    setb("log_thick", np.log10(0.01), np.log10(0.35))
    setb("log_pcs", np.log10(2.0), np.log10(60.0))
    setb("log_rch", np.log10(0.05), np.log10(10.0))
    setb("log_ghb", np.log10(1e-4), np.log10(1e-1))
    setb("grace_alpha", 0.75, 0.97)
    lo[LAYOUT["grace_drift"]] = np.array([-3.0, -20.0, -20.0])
    hi[LAYOUT["grace_drift"]] = np.array([3.0, 20.0, 20.0])
    setb("insar_ramp", -0.02, 0.02)
    return Prior(mean=mean, sd=sd, names=names, lo=lo, hi=hi)


def sample_prior(pr: Prior, ne: int, seed: int = 5) -> np.ndarray:
    """Draw `ne` prior realisations, shape (NPAR, ne).

    District abstraction is drawn with a three-year exponential temporal correlation,
    because an irrigation programme does not resample itself every January. Every
    other parameter is drawn independently.
    """
    rng = np.random.default_rng(seed)
    X = pr.mean[:, None] + pr.sd[:, None] * rng.standard_normal((NPAR, ne))

    yy = np.arange(C.NYEAR)
    cov = np.exp(-np.abs(yy[:, None] - yy[None, :]) / 3.0)
    L = np.linalg.cholesky(cov + 1e-8 * np.eye(C.NYEAR))
    sl = LAYOUT["logq"]
    q = pr.mean[sl].reshape(C.NDIST, C.NYEAR)[:, :, None] + 0.30 * np.einsum(
        "yz,dzn->dyn", L, rng.standard_normal((C.NDIST, C.NYEAR, ne)))
    X[sl] = q.reshape(C.NDIST * C.NYEAR, ne)

    return clip(X, pr)


def clip(X: np.ndarray, pr: Prior) -> np.ndarray:
    """Project an ensemble back into the admissible parameter box."""
    return np.clip(X, pr.lo[:, None], pr.hi[:, None])


# --------------------------------------------------------------------------- decoding
def q_annual(x: np.ndarray) -> np.ndarray:
    return 10.0 ** x[LAYOUT["logq"]].reshape(C.NDIST, C.NYEAR)


def to_params(x: np.ndarray, grid: C.Grid) -> M.Params:
    pts = fields.pilot_locations(grid, NPP)
    k1 = fields.pilot_to_grid(grid, pts, x[LAYOUT["logk1"]])
    k2 = fields.pilot_to_grid(grid, pts, x[LAYOUT["logk2"]])
    eta = x[LAYOUT["eta"]]
    pre = x[LAYOUT["preplant"]]
    mon = forcing.monthly_abstraction(q_annual(x), pre)
    g = lambda k: float(10.0 ** x[LAYOUT[k]][0])
    return M.Params(logk1=k1, logk2=k2, ss_cg=g("log_ss"), sy=g("log_sy"),
                    csub_ssv=g("log_ssv"), csub_sse=g("log_sse"),
                    thick_frac=g("log_thick"), pcs_offset=g("log_pcs"),
                    recharge_mm_yr=g("log_rch"), ghb_cond=g("log_ghb"),
                    q_monthly=mon, eta=eta)


# --------------------------------------------------------------------------- forward
def forward(x: np.ndarray, ws: Path, grid: C.Grid, mask: np.ndarray,
            geom: O.Geometry) -> dict[str, np.ndarray] | None:
    """Run one parameter vector and return its simulated observation vectors."""
    p = to_params(x, grid)
    M.build(ws, grid, p, mask)
    if not M.run(ws):
        return None
    st = O.read_state(ws, grid, sy=p.sy)
    return {
        "et": O.op_et(q_annual(x), x[LAYOUT["eta"]], x[LAYOUT["preplant"]]),
        "grace": O.op_grace(st, float(x[LAYOUT["grace_alpha"]][0]), x[LAYOUT["grace_drift"]]),
        "insar": O.op_insar(st, geom, grid, x[LAYOUT["insar_ramp"]]),
        "head": O.op_head(st, geom, grid),
        "meter": O.op_meter(q_annual(x)),
    }


# --------------------------------------------------------------------------- candidates
CAND_STEP_KM = 10.0
CAND_YEARS = (5, 10, 15, 20)


def candidate_cells(grid: C.Grid) -> np.ndarray:
    """Regular lattice of candidate instrument sites, shape (nc, 2) as (row, col)."""
    step = max(int(round(CAND_STEP_KM * 1000.0 / grid.delr_m)), 1)
    return np.array([[i, j] for i in range(step // 2, grid.nrow, step)
                     for j in range(step // 2, grid.ncol, step)])


def permanent_loss(inelastic: np.ndarray, grid: C.Grid, period: int = -1) -> float:
    """Storage capacity destroyed by inelastic compaction, m3.

    Read from the inelastic compaction field CSUB writes, so the number is the one
    the solver accounted for rather than a re-derivation of it.
    """
    return float(inelastic[period].sum() * grid.delr_m ** 2)


def forward_rich(x: np.ndarray, ws: Path, grid: C.Grid, mask: np.ndarray,
                 geom: O.Geometry) -> dict[str, np.ndarray] | None:
    """Forward run returning the installed observations, the candidate observations
    that could be installed, and the two forecasts the decision depends on."""
    p = to_params(x, grid)
    M.build(ws, grid, p, mask, inelastic=True)
    if not M.run(ws):
        return None
    st = O.read_state(ws, grid, sy=p.sy)
    inel = O.read_inelastic(ws, grid)
    qa = q_annual(x)
    cc = candidate_cells(grid)
    ti = [12 * y - 1 for y in CAND_YEARS]
    hcs = st.head[:, C.CSUB_LAYER]
    return {
        "et": O.op_et(qa, x[LAYOUT["eta"]], x[LAYOUT["preplant"]]),
        "grace": O.op_grace(st, float(x[LAYOUT["grace_alpha"]][0]), x[LAYOUT["grace_drift"]]),
        "insar": O.op_insar(st, geom, grid, x[LAYOUT["insar_ramp"]]),
        "head": O.op_head(st, geom, grid),
        "piezo": np.array([hcs[t][cc[:, 0], cc[:, 1]] for t in ti]).ravel(),
        "gnss": np.array([st.subsidence[t][cc[:, 0], cc[:, 1]] for t in ti]).ravel(),
        "meter": qa.ravel(),
        "fc_q_last5": np.array([qa[:, -5:].sum()]),
        "fc_perm_loss": np.array([permanent_loss(inel, grid)]),
    }
