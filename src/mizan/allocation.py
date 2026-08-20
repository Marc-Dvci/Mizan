"""Workstream C: risk-bounded allocation over a response-matrix surrogate.

The decision is a per-district annual quota over a twenty-year horizon. The quantity
being protected is not water in the ground, which can be refilled in principle, but
storage capacity, which cannot: once head passes the preconsolidation threshold the
skeleton compacts inelastically and that pore volume is gone at any price, including
the price of desalination.

The surrogate is a superposition response matrix built by pulsing each district in
turn, so head is linear in the quota vector. Inelastic compaction is the positive part
of the preconsolidation exceedance, which is convex in head, so the whole programme is
a linear programme in the Rockafellar-Uryasev form and the tail measure is exact rather
than sampled.

Resolution matters here and is not a free parameter. Compaction is a positive part, so
averaging head over an area before applying it understates the loss by Jensen's
inequality. A first version of this surrogate worked at district resolution and
predicted 723 Mm3 of permanent loss where the simulator produced 3,441 Mm3, and the
policy it selected came out worse than the baseline when run. The surrogate therefore
works on drawdown-stratified zones inside each district, and every candidate policy is
still re-run in full MODFLOW with the inelastic switch active.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config as C
from . import estimator as E
from . import forcing, model as M, observations as O

HORIZON_Y = 20
NZONE = 5                      # drawdown strata per district


# --------------------------------------------------------------------------- zones
def zone_map(grid: C.Grid, drawdown: np.ndarray, nzone: int = NZONE) -> np.ndarray:
    """Zone index for every cell, shape (nrow, ncol).

    Each district is split into `nzone` strata of equal cell count, ordered by the
    drawdown it has already taken. Within a stratum the head is close to uniform, so
    the positive part can be applied to the stratum mean without losing the cells that
    are already deep in the inelastic regime.
    """
    dmap = grid.district_map()
    out = np.zeros((grid.nrow, grid.ncol), dtype=int)
    for d in range(C.NDIST):
        sel = dmap == d
        v = drawdown[sel]
        q = np.quantile(v, np.linspace(0.0, 1.0, nzone + 1)[1:-1])
        out[sel] = d * nzone + np.searchsorted(q, v)
    return out


def zone_cells(zmap: np.ndarray, nzone: int = NZONE) -> list[np.ndarray]:
    return [np.argwhere(zmap == z) for z in range(C.NDIST * nzone)]


def zone_head(head: np.ndarray, cells: list[np.ndarray]) -> np.ndarray:
    """Annual mean head in the pumped layer, per zone, shape (nzones, nyear)."""
    ny = head.shape[0] // 12
    h = head[: ny * 12, C.CSUB_LAYER]
    out = np.zeros((len(cells), ny))
    for z, cc in enumerate(cells):
        if len(cc):
            out[z] = h[:, cc[:, 0], cc[:, 1]].mean(axis=1).reshape(ny, 12).mean(axis=1)
    return out


def build_future_q(q_hist: np.ndarray, q_future: np.ndarray,
                   preplant: np.ndarray) -> np.ndarray:
    """Monthly rates over history followed by horizon, m3/d."""
    return forcing.monthly_abstraction(np.concatenate([q_hist, q_future], axis=1),
                                       preplant)


@dataclass
class MemberSurrogate:
    h_ref: np.ndarray        # (NZ, HORIZON_Y) zone head under the reference policy
    R: np.ndarray            # (NZ, NDIST, HORIZON_Y) head response per m3/yr, by lag
    pcs: np.ndarray          # (NZ,) preconsolidation head at the end of history
    ssv_b: float             # inelastic storage times interbed thickness, m per m
    area: np.ndarray         # (NZ,) zone area, m2
    loss_hist: float         # permanent loss already taken over the history, m3


def _run(x, ws, grid, mask, q_future, inelastic: bool = False):
    p = E.to_params(x, grid)
    p.q_monthly = build_future_q(E.q_annual(x), q_future, x[E.LAYOUT["preplant"]])
    M.build(ws, grid, p, mask, inelastic=inelastic)
    if not M.run(ws):
        return None
    return O.read_state(ws, grid, sy=p.sy), p


def member_surrogate(x, ws: Path, grid, mask, q_ref: np.ndarray,
                     delta_frac: float = 0.15):
    """Reference trajectory and superposition response matrix for one member.

    Costs one reference run plus one pulse run per district. The pulse is applied in
    the first horizon year only; time invariance then fills every later lag, and that
    assumption is measured by `lti_check` rather than assumed.
    """
    r = _run(x, ws, grid, mask, q_ref, inelastic=True)
    if r is None:
        return None
    st, p = r
    hist = st.head[:C.NPER, C.CSUB_LAYER]
    cells = zone_cells(zone_map(grid, C.H_INIT - hist[-1]))
    area = np.array([len(c) * grid.delr_m ** 2 for c in cells], dtype=float)

    h_ref = zone_head(st.head, cells)[:, C.NYEAR:]
    pcs_field = np.minimum(C.H_INIT - p.pcs_offset, hist.min(axis=0))
    pcs = np.array([pcs_field[c[:, 0], c[:, 1]].mean() if len(c) else 0.0 for c in cells])
    loss_hist = float(O.read_inelastic(ws, grid)[C.NPER - 1].sum() * grid.delr_m ** 2)

    R = np.zeros((len(cells), C.NDIST, HORIZON_Y))
    for d in range(C.NDIST):
        dq = delta_frac * max(q_ref[d, 0], 1.0)
        qp = q_ref.copy()
        qp[d, 0] += dq
        rp = _run(x, ws, grid, mask, qp)
        if rp is None:
            return None
        hp = zone_head(rp[0].head, cells)[:, C.NYEAR:]
        R[:, d, :] = (hp - h_ref) / dq

    ssv_b = p.csub_ssv * p.thick_frac * (C.BOTM[0] - C.BOTM[1])
    return MemberSurrogate(h_ref=h_ref, R=R, pcs=pcs, ssv_b=ssv_b, area=area,
                           loss_hist=loss_hist)


def lti_check(x, ws, grid, mask, q_ref, surr, pulse_year: int = 10,
              delta_frac: float = 0.15) -> dict:
    """Measure the time invariance the response matrix assumes.

    The matrix is built from a pulse in the first horizon year and reused at every
    later lag. Pulsing the same district in a later year and comparing against what
    superposition predicts gives the error that assumption carries. The aquifer is not
    exactly linear: the upper unit is unconfined and inelastic compaction switches on
    at a threshold.
    """
    r0 = _run(x, ws, grid, mask, q_ref)
    if r0 is None:
        return {"max_rel_error": float("nan"), "mean_rel_error": float("nan")}
    hist = r0[0].head[:C.NPER, C.CSUB_LAYER]
    cells = zone_cells(zone_map(grid, C.H_INIT - hist[-1]))

    out = {}
    for d in range(C.NDIST):
        dq = delta_frac * max(q_ref[d, 0], 1.0)
        qp = q_ref.copy()
        qp[d, pulse_year] += dq
        r = _run(x, ws, grid, mask, qp)
        if r is None:
            continue
        got = zone_head(r[0].head, cells)[:, C.NYEAR:] - surr.h_ref
        pred = np.zeros_like(got)
        for t in range(pulse_year, HORIZON_Y):
            pred[:, t] = surr.R[:, d, t - pulse_year] * dq
        den = np.abs(pred).max()
        out[f"d{d}"] = float(np.abs(got - pred).max() / den) if den > 0 else float("nan")
    vals = [v for v in out.values() if np.isfinite(v)]
    return {"per_district": out,
            "max_rel_error": float(max(vals)) if vals else float("nan"),
            "mean_rel_error": float(np.mean(vals)) if vals else float("nan")}


def head_from_policy(s: MemberSurrogate, q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Zone head trajectory under quota `q`, by superposition, shape (NZ, H)."""
    h = s.h_ref.copy()
    dq = q - q_ref
    for t in range(HORIZON_Y):
        for lag in range(t + 1):
            h[:, t] += s.R[:, :, t - lag] @ dq[:, lag]
    return h


def response_operator(s: MemberSurrogate) -> np.ndarray:
    """Flattened operator L with h = h_ref + L (q - q_ref), shape (NZ*H, NDIST*H)."""
    nz, n, H = s.R.shape[0], C.NDIST, HORIZON_Y
    L = np.zeros((nz * H, n * H))
    for t in range(H):
        for sdx in range(t + 1):
            L[t * nz:(t + 1) * nz, sdx * n:(sdx + 1) * n] = s.R[:, :, t - sdx]
    return L


# --------------------------------------------------------------------------- policies
def uniform_policy(q_hist: np.ndarray, total: float) -> np.ndarray:
    """The instrument regulators actually use: every district keeps its historical
    share and takes the same proportional cut, held flat across the horizon."""
    share = q_hist[:, -3:].mean(axis=1)
    share = share / share.sum()
    return np.outer(share, np.full(HORIZON_Y, total / HORIZON_Y))


def evaluate_policy(surrs, q: np.ndarray, q_ref: np.ndarray) -> dict:
    """Surrogate prediction of permanent loss and exceedance for a quota vector."""
    loss, worst = [], []
    for s in surrs:
        h = head_from_policy(s, q, q_ref)
        g = (s.pcs[:, None] - h).max(axis=1)
        worst.append(float(g.max()))
        loss.append(float(s.ssv_b * (s.area * np.maximum(g, 0.0)).sum()))
    loss = np.asarray(loss)
    worst = np.asarray(worst)
    return {
        "loss_mean_mcm": float(loss.mean() / 1e6),
        "loss_cvar90_mcm": float(np.sort(loss)[int(0.9 * len(loss)):].mean() / 1e6),
        "loss_p90_mcm": float(np.quantile(loss, 0.9) / 1e6),
        "p_exceed": float((worst > 0).mean()),
        "worst_exceed_m": float(worst.max()),
    }


QSCALE = 1.0e6            # decision variables are solved in Mm3, not m3


def optimise(surrs, q_ref: np.ndarray, total: float, cap: np.ndarray,
             floor: np.ndarray, beta: float = 0.90,
             chance: float | None = None,
             solvers=("CLARABEL", "SCS", "ECOS", "OSQP")):
    """Minimise the conditional value at risk of permanent storage loss.

    Volumes are solved in Mm3 and responses scaled to match. Left in cubic metres the
    programme spans fifteen orders of magnitude between a quota and a head response and
    no interior-point solver will take it.
    """
    import cvxpy as cp

    n, H = C.NDIST, HORIZON_Y
    nz = surrs[0].R.shape[0]
    m = len(surrs)
    qs = cp.Variable(n * H, nonneg=True)                 # Mm3/yr
    qref_s = q_ref.flatten(order="F") / QSCALE

    losses, worsts = [], []
    for s in surrs:
        L = response_operator(s) * QSCALE                # m per Mm3/yr
        hf = s.h_ref.flatten(order="F") + L @ (qs - qref_s)
        exc = cp.hstack([cp.max(s.pcs[z] - hf[z::nz]) for z in range(nz)])
        worsts.append(cp.max(exc))
        losses.append((s.ssv_b / QSCALE) * (s.area @ cp.pos(exc)))
    loss = cp.hstack(losses)

    tau = cp.Variable()
    z = cp.Variable(m, nonneg=True)
    cons = [z >= loss - tau,
            cp.sum(qs) == total / QSCALE,
            qs <= cap.flatten(order="F") / QSCALE]
    for d in range(n):
        cons.append(cp.sum(qs[d::n]) >= floor[d] / QSCALE)
    obj = tau + cp.sum(z) / ((1.0 - beta) * m)

    if chance is not None:
        w = cp.hstack(worsts)
        tc = cp.Variable()
        zc = cp.Variable(m, nonneg=True)
        cons += [zc >= w - tc, tc + cp.sum(zc) / ((1.0 - chance) * m) <= 0.0]

    prob = cp.Problem(cp.Minimize(obj), cons)
    last = "not attempted"
    for sv in solvers:
        try:
            prob.solve(solver=sv)
        except Exception as exc:
            last = f"{sv}: {exc}"
            continue
        if qs.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
            return np.asarray(qs.value).reshape(H, n).T * QSCALE, f"{sv}/{prob.status}"
        last = f"{sv}: {prob.status}"
    return None, last
