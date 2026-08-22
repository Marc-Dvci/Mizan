"""Ensemble smoother with multiple data assimilation, and the localisation that makes
it usable at this ensemble size.

Localisation here is mostly exact rather than heuristic. An evapotranspiration
retrieval over district d in year y carries no information about the conductivity
field, and no observation carries information about pumping that had not happened
when it was taken. Those zeros are written into the taper directly. Only the head and
interferometric blocks for pumping and pilot points use a distance taper.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import config as C
from . import estimator as E
from . import fields
from . import observations as O

LEGS = ("et", "grace", "insar", "head", "meter")

_W: dict = {}


def _init(root, grid, mask, geom):
    _W["root"] = Path(root)
    _W["grid"] = grid
    _W["mask"] = mask
    _W["geom"] = geom
    _W["ws"] = Path(root) / f"w{os.getpid()}"


def _run_one(arg):
    i, x = arg
    try:
        r = E.forward(x, _W["ws"], _W["grid"], _W["mask"], _W["geom"])
    except Exception:
        r = None
    return i, r


def run_ensemble(X, root, grid, mask, geom, legs=LEGS, workers=6):
    """Run every column of X. Returns (D, ok) with D of shape (nobs, ne)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ne = X.shape[1]
    out = [None] * ne
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(str(root), grid, mask, geom)) as ex:
        for i, r in ex.map(_run_one, [(i, X[:, i]) for i in range(ne)], chunksize=1):
            out[i] = r
    ok = np.array([r is not None for r in out])
    if not ok.any():
        raise RuntimeError("every ensemble member failed")
    ref = next(r for r in out if r is not None)
    fill = np.concatenate([ref[k] for k in legs])
    D = np.zeros((fill.size, ne))
    for i, r in enumerate(out):
        D[:, i] = np.concatenate([r[k] for k in legs]) if r is not None else fill
    return D, ok


# --------------------------------------------------------------------------- taper
def _gc(d, c):
    """Gaspari-Cohn fifth-order piecewise correlation, zero beyond twice c."""
    r = np.abs(d) / c
    out = np.zeros_like(r)
    m = r <= 1
    x = r[m]
    out[m] = -0.25 * x ** 5 + 0.5 * x ** 4 + 0.625 * x ** 3 - 5.0 / 3.0 * x ** 2 + 1.0
    m = (r > 1) & (r <= 2)
    x = np.maximum(r[m], 1e-6)
    out[m] = (x ** 5 / 12.0 - 0.5 * x ** 4 + 0.625 * x ** 3 + 5.0 / 3.0 * x ** 2
              - 5.0 * x + 4.0 - 2.0 / (3.0 * x))
    return np.clip(out, 0.0, 1.0)


def district_centres():
    third = C.DOMAIN_KM * 1000.0 / 3.0
    dcx = np.array([((d % 3) + 0.5) * third for d in range(C.NDIST)])
    dcy = np.array([((d // 3) + 0.5) * third for d in range(C.NDIST)])
    return dcx, dcy


def obs_meta(geom, legs=LEGS):
    """Leg index, time index, location and district for every observation."""
    leg, t, x, y, dist = [], [], [], [], []
    dcx, dcy = district_centres()
    for name in legs:
        if name == "et":
            for d in range(C.NDIST):
                for k in range(C.NPER):
                    leg.append(0); t.append(k); x.append(dcx[d]); y.append(dcy[d]); dist.append(d)
        elif name == "grace":
            for k in range(C.NPER):
                leg.append(1); t.append(k); x.append(np.nan); y.append(np.nan); dist.append(-1)
        elif name == "insar":
            for k in geom.insar_epochs:
                for px in geom.insar_xy:
                    leg.append(2); t.append(int(k)); x.append(px[0]); y.append(px[1]); dist.append(-1)
        elif name == "head":
            rows, cols = np.nonzero(geom.well_seen)
            for w, k in zip(rows, cols):
                leg.append(3); t.append(int(k)); x.append(geom.well_xy[w, 0])
                y.append(geom.well_xy[w, 1]); dist.append(-1)
        elif name == "meter":
            for d in range(C.NDIST):
                for yv in range(C.NYEAR):
                    leg.append(4); t.append(yv * 12 + 6)
                    x.append(dcx[d]); y.append(dcy[d]); dist.append(d)
    return {"leg": np.array(leg), "t": np.array(t), "x": np.array(x, dtype=float),
            "y": np.array(y, dtype=float), "dist": np.array(dist)}


def taper(geom, grid, legs=LEGS, radius_km=40.0):
    """Localisation matrix of shape (NPAR, NOBS)."""
    m = obs_meta(geom, legs)
    nobs = m["leg"].size
    rho = np.zeros((E.NPAR, nobs), dtype=np.float32)
    is_et = m["leg"] == 0
    is_gr = m["leg"] == 1
    is_in = m["leg"] == 2
    is_hd = m["leg"] == 3
    is_mt = m["leg"] == 4
    geo = is_in | is_hd
    yobs = m["t"] // 12
    dcx, dcy = district_centres()

    sl = E.LAYOUT["logq"]
    for d in range(C.NDIST):
        wgeo = _gc(np.hypot(m["x"][geo] - dcx[d], m["y"][geo] - dcy[d]) / 1000.0, radius_km)
        for yv in range(C.NYEAR):
            v = np.zeros(nobs, dtype=np.float32)
            v[is_et] = ((m["dist"][is_et] == d) & (yobs[is_et] == yv)).astype(np.float32)
            v[is_mt] = ((m["dist"][is_mt] == d) & (yobs[is_mt] == yv)).astype(np.float32)
            v[is_gr] = (yobs[is_gr] >= yv).astype(np.float32)
            v[geo] = wgeo * (yobs[geo] >= yv)
            rho[sl.start + d * C.NYEAR + yv] = v

    for key in ("eta", "preplant"):
        sl = E.LAYOUT[key]
        for d in range(C.NDIST):
            v = np.zeros(nobs, dtype=np.float32)
            v[is_et] = (m["dist"][is_et] == d).astype(np.float32)
            v[is_mt] = (m["dist"][is_mt] == d).astype(np.float32)
            v[is_gr] = 1.0
            v[geo] = _gc(np.hypot(m["x"][geo] - dcx[d], m["y"][geo] - dcy[d]) / 1000.0, radius_km)
            rho[sl.start + d] = v

    pts = fields.pilot_locations(grid, E.NPP)
    for key in ("logk1", "logk2"):
        sl = E.LAYOUT[key]
        for i in range(len(pts)):
            v = np.zeros(nobs, dtype=np.float32)
            v[is_gr] = 1.0
            v[geo] = _gc(np.hypot(m["x"][geo] - pts[i, 0], m["y"][geo] - pts[i, 1]) / 1000.0,
                         radius_km + 15.0)
            rho[sl.start + i] = v

    for key in ("log_ss", "log_sy", "log_ssv", "log_sse", "log_thick", "log_pcs",
                "log_rch", "log_ghb"):
        rho[E.LAYOUT[key]] = (is_gr | geo).astype(np.float32)
    rho[E.LAYOUT["grace_alpha"]] = is_gr.astype(np.float32)
    rho[E.LAYOUT["grace_drift"]] = is_gr.astype(np.float32)
    rho[E.LAYOUT["insar_ramp"]] = is_in.astype(np.float32)
    return rho


# --------------------------------------------------------------------------- ES-MDA
def alpha_schedule(na: int, ratio: float = 2.0) -> np.ndarray:
    """Decreasing inflation coefficients with the reciprocals summing to one.

    The first assimilation is heavily damped, because a prior drawn without reference
    to the data sits tens of standard deviations from it and an undamped first step
    would leave the physical parameter box entirely.
    """
    a = ratio ** np.arange(na)
    return a.sum() / ratio ** np.arange(na)


def _groups(rho: np.ndarray) -> dict:
    """Parameters sharing an identical localisation vector, so each mask is built once."""
    out: dict = {}
    for r in range(rho.shape[0]):
        out.setdefault(rho[r].tobytes(), []).append(r)
    return out


def esmda_update(X, D, d_obs, sd, alpha, rho, rng, ok=None, pr=None, max_step=2.0,
                 rho_floor=1e-2, rtps=0.7):
    """One ES-MDA analysis step with localised local analysis.

    Localisation is applied to the observation error covariance rather than to the
    cross-covariance alone. Each parameter group is updated from the observations its
    taper admits, with the error variance of every admitted observation divided by its
    taper weight, which is the R-localisation of the local ensemble transform filter.
    Tapering the cross-covariance on its own leaves the analysis inconsistent with the
    innovation covariance it was solved against, and diverges when the prior sits far
    from the data.

    The gain is taken in the ensemble subspace through the ne-by-ne normal equations,
    so no matrix the size of the observation vector is ever formed.

    `rtps` is relaxation to prior spread. A localised smoother iterated on a problem
    with strong degeneracies under-disperses, and an interval that does not cover is
    worse than no interval, because any posterior decision made from it would be
    overconfident. After each analysis the posterior anomalies are
    relaxed back towards the prior spread by this fraction, which is the standard
    treatment and costs nothing in the mean.
    """
    ne = X.shape[1]
    use = np.ones(ne, dtype=bool) if ok is None else ok
    Xa, Da = X[:, use], D[:, use]
    nu = Xa.shape[1]

    dX = Xa - Xa.mean(axis=1, keepdims=True)
    dD = Da - Da.mean(axis=1, keepdims=True)
    noise = rng.standard_normal((d_obs.size, nu))

    if rho is None:
        rho = np.ones((X.shape[0], d_obs.size), dtype=np.float32)

    step = np.zeros((X.shape[0], nu))
    for _, rows in _groups(rho).items():
        w = rho[rows[0]]
        m = w > rho_floor
        if not m.any():
            continue
        sde = sd[m] / np.sqrt(w[m])
        S = (dD[m] / sde[:, None]) / np.sqrt(nu - 1)
        R = (d_obs[m, None] + np.sqrt(alpha) * sde[:, None] * noise[m] - Da[m]) / sde[:, None]
        G = S.T @ S
        G[np.diag_indices_from(G)] += alpha
        step[rows] = dX[rows] @ np.linalg.solve(G, S.T @ R) / np.sqrt(nu - 1)

    if pr is not None:
        cap = max_step * pr.sd[:, None]
        step = np.clip(step, -cap, cap)
    Xn = X.copy()
    Xn[:, use] = Xa + step
    if rtps > 0.0:
        sd_pri = dX.std(axis=1)
        an = Xn[:, use] - Xn[:, use].mean(axis=1, keepdims=True)
        sd_post = an.std(axis=1)
        good = sd_post > 1e-12
        lam = np.ones_like(sd_post)
        lam[good] = 1.0 + rtps * (sd_pri[good] - sd_post[good]) / sd_post[good]
        lam = np.clip(lam, 1.0, 25.0)
        Xn[:, use] = Xn[:, use].mean(axis=1, keepdims=True) + lam[:, None] * an
    if (~use).any():
        pick = rng.choice(np.nonzero(use)[0], size=int((~use).sum()))
        Xn[:, ~use] = Xn[:, pick]
    return E.clip(Xn, pr) if pr is not None else Xn


# --------------------------------------------------------------------------- error budget
def _lag1(res: np.ndarray, site: np.ndarray, time: np.ndarray) -> float:
    """Mean lag-one autocorrelation of a residual, taken within each site."""
    vals = []
    for s in np.unique(site):
        m = site == s
        r = res[m][np.argsort(time[m])]
        if r.size > 8 and r.std() > 0:
            v = np.corrcoef(r[:-1], r[1:])[0, 1]
            if np.isfinite(v):
                vals.append(v)
    return float(np.clip(np.mean(vals), 0.0, 0.95)) if vals else 0.0


def total_error(sim_mean: np.ndarray, d_obs: np.ndarray, sd_nom: np.ndarray,
                geom, legs=LEGS) -> tuple[np.ndarray, dict]:
    """Observation error inflated to cover model structural error.

    Two effects are estimated from the residual of a first-stage inversion, without
    reference to the truth. The residual root mean square in excess of the nominal
    instrument error is structural: the coarse grid and the pilot-point basis cannot
    reproduce the fine field that generated the data. The residual autocorrelation
    within a site says how many of those observations are independent; treating a
    persistent bias at a well as two hundred independent measurements is what collapses
    an ensemble and produces intervals that do not cover.
    """
    m = obs_meta(geom, legs)
    site = np.where(np.isnan(m["x"]), -1.0, m["x"] * 1e6 + m["y"])
    res = sim_mean - d_obs
    sd = sd_nom.copy()
    report = {}
    for name in legs:
        sel = m["leg"] == LEGS.index(name)
        if not sel.any():
            continue
        rms = float(np.sqrt((res[sel] ** 2).mean()))
        nom = float(np.sqrt((sd_nom[sel] ** 2).mean()))
        struct = float(np.sqrt(max(rms ** 2 - nom ** 2, 0.0)))
        r1 = _lag1(res[sel], site[sel], m["t"][sel])
        infl = float(np.sqrt((1.0 + r1) / (1.0 - r1)))
        sd[sel] = np.sqrt(sd_nom[sel] ** 2 + struct ** 2) * infl
        report[name] = {"nominal": nom, "residual_rms": rms, "structural": struct,
                        "lag1": r1, "independence_inflation": infl,
                        "total": float(np.sqrt((sd[sel] ** 2).mean()))}
    return sd, report
