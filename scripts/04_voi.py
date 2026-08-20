"""Workstream C, second half: where the next meter goes.

A finite-difference Jacobian is taken at the posterior mean over every parameter, for
the observations already installed and for every instrument that could be installed.
Linear data-worth analysis then ranks candidate instruments by the variance they
remove from two forecasts: recent basin abstraction, and the storage capacity that
inelastic compaction destroys permanently.

The Schur complement used here is cross-checked against pyEMU on the same inputs.

Usage:  python scripts/04_voi.py [--nmeters 20]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mizan import config as C, estimator as E, fields, forcing as F
from mizan import observations as O, voi as V

RUNS = ROOT / "runs" / "jac"
LEGS_INSTALLED = ("et", "grace", "insar", "head")
CANDIDATE_LEGS = ("meter", "piezo", "gnss")
FORECASTS = ("fc_q_last5", "fc_perm_loss")
_W: dict = {}


def _init(mask, geom):
    _W["mask"] = mask
    _W["geom"] = geom
    _W["ws"] = RUNS / f"w{os.getpid()}"


def _run(arg):
    i, x = arg
    try:
        return i, E.forward_rich(x, _W["ws"], C.EST, _W["mask"], _W["geom"])
    except Exception:
        return i, None


def flatten(r, keys):
    return np.concatenate([r[k] for k in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmeters", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    tr = np.load(ROOT / "results" / "truth.npz")
    post = np.load(ROOT / "results" / "posterior_H.npz")
    X, ok = post["X"], post["ok"]
    x0 = X[:, ok].mean(axis=1)
    mask_e = fields.upscale_mask(F.pivot_mask(C.TRUTH), 2)
    geom = O.Geometry(well_xy=tr["well_xy"], well_seen=tr["well_seen"],
                      insar_xy=tr["insar_xy"], insar_ref_xy=tr["insar_ref_xy"],
                      insar_epochs=np.arange(C.INSAR_STACK_MONTHS - 1, C.NPER,
                                             C.INSAR_STACK_MONTHS))
    pr = E.prior(mask_e, C.EST)

    step = 0.05 * pr.sd
    jobs = [(0, x0)] + [(k + 1, x0 + step[k] * np.eye(E.NPAR)[k]) for k in range(E.NPAR)]
    t0 = time.time()
    out = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(mask_e, geom)) as ex:
        for i, r in ex.map(_run, jobs, chunksize=1):
            out[i] = r
    print(f"Jacobian: {len(jobs)} runs in {time.time()-t0:.0f}s, "
          f"{sum(r is None for r in out)} failed")
    if out[0] is None:
        raise SystemExit("base run failed")

    keys = list(LEGS_INSTALLED) + list(CANDIDATE_LEGS) + list(FORECASTS)
    base = flatten(out[0], keys)
    J = np.zeros((base.size, E.NPAR))
    good = np.ones(E.NPAR, dtype=bool)
    for k in range(E.NPAR):
        r = out[k + 1]
        if r is None:
            good[k] = False
            continue
        J[:, k] = (flatten(r, keys) - base) / step[k]

    off, idx = 0, {}
    for k in keys:
        n = out[0][k].size
        idx[k] = slice(off, off + n)
        off += n

    # the installed legs carry the same total error the posterior was built with,
    # not their nominal instrument error, or the data worth of a new instrument is
    # measured against a likelihood the inference never used
    budget = np.load(ROOT / "results" / "error_budget.npy")
    off4, sd4 = 0, {}
    for k in ("et", "grace", "insar", "head", "meter"):
        n = tr["obs_" + k].size
        sd4[k] = budget[off4:off4 + n]
        off4 += n
    sigma = np.zeros(base.size)
    for k in ("et", "grace", "insar", "head"):
        sigma[idx[k]] = sd4[k]
    sigma[idx["meter"]] = np.maximum(O.METER_REL_SIGMA * np.abs(out[0]["meter"]), 1.0e5)
    sigma[idx["piezo"]] = 0.30
    sigma[idx["gnss"]] = 0.003
    sigma[idx["fc_q_last5"]] = 1.0
    sigma[idx["fc_perm_loss"]] = 1.0

    dw = V.DataWorth(jac=J, sigma_obs=sigma, sigma_par=pr.sd, names=keys)
    installed = np.zeros(base.size, dtype=bool)
    for k in LEGS_INSTALLED:
        installed[idx[k]] = True

    cc = E.candidate_cells(C.EST)
    nt = len(E.CAND_YEARS)
    cands: dict[str, np.ndarray] = {}
    for d in range(C.NDIST):
        m = np.zeros(base.size, dtype=bool)
        m[idx["meter"].start + d * C.NYEAR: idx["meter"].start + (d + 1) * C.NYEAR] = True
        cands[f"meter_d{d}"] = m
    for kind in ("piezo", "gnss"):
        for j in range(len(cc)):
            m = np.zeros(base.size, dtype=bool)
            for t in range(nt):
                m[idx[kind].start + t * len(cc) + j] = True
            cands[f"{kind}_{j}"] = m

    res = {}
    for fname in FORECASTS:
        f = J[idx[fname].start]
        s_prior = float(np.sqrt(f @ np.diag(pr.sd ** 2) @ f))
        s_post = dw.forecast_sd(installed, f)
        worth = dw.worth(installed, cands, f)
        order, curve = dw.greedy(installed, cands, f, args.nmeters)
        top = sorted(worth.items(), key=lambda kv: -kv[1])[:12]
        res[fname] = {
            "prior_sd": s_prior, "posterior_sd": s_post,
            "top_candidates": [{"name": k, "sd_removed": v} for k, v in top],
            "greedy_order": order, "greedy_curve": curve,
        }
        print(f"\n{fname}: prior sd {s_prior:.4g} -> posterior {s_post:.4g}")
        for k, v in top[:6]:
            print(f"   {k:14s} removes {v:.4g} ({v/s_post*100:5.1f}% of what is left)")
        print(f"   {args.nmeters} instruments take it to {curve[-1]:.4g} "
              f"({(1-curve[-1]/s_post)*100:.1f}% reduction)")

    res["pyemu_check"] = pyemu_cross_check(J, sigma, pr, idx, installed, FORECASTS)
    res["n_failed_columns"] = int((~good).sum())
    np.savez_compressed(ROOT / "results" / "voi.npz", J=J, sigma=sigma,
                        cand_cells=cc, installed=installed)
    (ROOT / "results" / "voi.json").write_text(json.dumps(res, indent=2))
    print("\nwrote results/voi.json")


def pyemu_cross_check(J, sigma, pr, idx, installed, forecasts):
    """Recompute the posterior forecast variance with pyEMU and compare."""
    try:
        import pandas as pd
        import pyemu
        rows = np.nonzero(installed)[0]
        onames = [f"o{i}" for i in rows] + list(forecasts)
        pnames = [f"p{i}" for i in range(J.shape[1])]
        sub = np.vstack([J[rows], np.array([J[idx[f].start] for f in forecasts])])
        jco = pyemu.Jco.from_dataframe(pd.DataFrame(sub, index=onames, columns=pnames))
        parcov = pyemu.Cov(x=np.atleast_2d(pr.sd ** 2).T, names=pnames, isdiagonal=True)
        ovar = np.concatenate([sigma[rows] ** 2, np.ones(len(forecasts))])
        obscov = pyemu.Cov(x=np.atleast_2d(ovar).T, names=onames, isdiagonal=True)
        pst = pyemu.Pst.from_par_obs_names(par_names=pnames, obs_names=onames)
        sc = pyemu.Schur(jco=jco, pst=pst, parcov=parcov, obscov=obscov,
                         forecasts=list(forecasts))
        theirs = {k: float(np.sqrt(v)) for k, v in sc.posterior_forecast.items()}
        dw = V.DataWorth(jac=J, sigma_obs=sigma, sigma_par=pr.sd, names=[])
        mine = {f: dw.forecast_sd(installed, J[idx[f].start]) for f in forecasts}
        out = {f: {"mizan": mine[f], "pyemu": theirs.get(f),
                   "rel_diff": abs(mine[f] - theirs.get(f, np.nan)) / max(mine[f], 1e-30)}
               for f in forecasts}
        for f, v in out.items():
            print(f"pyEMU cross-check {f}: mizan {v['mizan']:.6g} "
                  f"pyemu {v['pyemu']:.6g} rel diff {v['rel_diff']:.2e}")
        return out
    except Exception as exc:                                  # pragma: no cover
        print(f"pyEMU cross-check unavailable: {exc}")
        return {"error": str(exc)}


if __name__ == "__main__":
    main()
