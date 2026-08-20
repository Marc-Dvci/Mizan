"""Workstream A: the L0 ablation grid.

Nine inversions of the same synthetic aquifer, differing only in which observations
the estimator is allowed to see. The scored quantity is district-annual abstraction
against the withheld truth. Two open-loop baselines are scored on the same target:
the published fixed-efficiency account, and the best that form can do with a global
efficiency fitted against the answer.

Usage:  python scripts/01_ablation.py [--ne 250] [--na 4] [--rows H,F,...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mizan import config as C, forcing as F, fields, estimator as E
from mizan import inversion as I, metrics as MT, observations as O

ROWS = {
    "A": ("head",),
    "B": ("grace",),
    "C": ("insar",),
    "D": ("et", "head"),
    "E": ("et", "grace"),
    "F": ("et", "insar"),
    "G": ("head", "grace", "insar"),
    "H": ("et", "grace", "insar", "head"),
    "ET": ("et",),
    "M": ("meter",),
    "HM1": ("et", "grace", "insar", "head", "meter"),
    "HM3": ("et", "grace", "insar", "head", "meter"),
    "HS": ("et", "grace", "insar", "head"),
    "HS3": ("et", "grace", "insar", "head"),
    "SAT": ("et", "grace", "insar"),
}
# Districts whose abstraction is metered, per row. Empty means the whole leg.
ROW_METERS = {"M": None, "HM1": [4], "HM3": [4, 2, 7]}
# Rows restricted to a sparse monitoring network. The target basin has almost no
# observation wells, so the configuration that matters there is not the one with a
# well every hundred square kilometres.
ROW_WELLS = {"HS": 10, "HS3": 3}
ROW_LABEL = {
    "ET": "evapotranspiration only, closure",
    "A": "heads only",
    "B": "gravity only",
    "C": "deformation only",
    "D": "evapotranspiration + heads",
    "E": "evapotranspiration + gravity",
    "F": "evapotranspiration + deformation",
    "G": "heads + gravity + deformation",
    "H": "all four, coupled closure",
    "M": "meters on every district",
    "HM1": "all four, plus one metered district",
    "HM3": "all four, plus three metered districts",
    "HS": "all four, 10 wells instead of 97",
    "HS3": "all four, 3 wells instead of 97",
    "SAT": "satellites only, no wells at all",
}


def load_setup(truth="truth.npz"):
    tr = np.load(ROOT / "results" / truth)
    mask_t = F.pivot_mask(C.TRUTH)
    mask_e = fields.upscale_mask(mask_t, int(C.EST.delr_m // C.TRUTH.delr_m))
    geom = O.Geometry(well_xy=tr["well_xy"], well_seen=tr["well_seen"],
                      insar_xy=tr["insar_xy"], insar_ref_xy=tr["insar_ref_xy"],
                      insar_epochs=np.arange(C.INSAR_STACK_MONTHS - 1, C.NPER,
                                             C.INSAR_STACK_MONTHS))
    return tr, mask_t, mask_e, geom


def leg_index(tr):
    idx, off = {}, 0
    for k in I.LEGS:
        n = tr["obs_" + k].size
        idx[k] = slice(off, off + n)
        off += n
    return idx


def head_subset(tr, nwell):
    """Flat indices inside the head block belonging to the first `nwell` wells."""
    seen = tr["well_seen"]
    flat = np.zeros(seen.shape, dtype=int)
    flat[seen] = np.arange(int(seen.sum()))
    keep = []
    for w in range(min(nwell, seen.shape[0])):
        keep.append(flat[w][seen[w]])
    return np.concatenate(keep) if keep else np.zeros(0, dtype=int)


def run_row(legs, X0, D0, ok0, tr, mask_e, geom, na, workers, root, pr, sd_full,
            meters=None, wells=None, rtps=0.7, seed=5):
    """One ES-MDA inversion restricted to `legs`. The prior ensemble is shared.

    `meters` restricts the metering leg to a list of districts, which is how the grid
    answers what a small number of installed meters is worth on top of the satellites.
    """
    idx = leg_index(tr)
    parts, sub_local = [], []
    off_local = 0
    for k in legs:
        if k == "meter" and meters is not None:
            for d in meters:
                a = idx[k].start + d * C.NYEAR
                parts.append(np.arange(a, a + C.NYEAR))
                sub_local.append(off_local + d * C.NYEAR + np.arange(C.NYEAR))
        elif k == "head" and wells is not None:
            h = head_subset(tr, wells)
            parts.append(idx[k].start + h)
            sub_local.append(off_local + h)
        else:
            n = idx[k].stop - idx[k].start
            parts.append(np.arange(idx[k].start, idx[k].stop))
            sub_local.append(off_local + np.arange(n))
        off_local += idx[k].stop - idx[k].start
    take = np.concatenate(parts)
    obs_full = np.concatenate([tr["obs_" + k] for k in I.LEGS])
    sel_obs = obs_full[take]
    sel_sd = sd_full[take]
    rho = I.taper(geom, C.EST, legs=legs)
    rho = rho[:, np.concatenate(sub_local)]

    rng = np.random.default_rng(97 + 1000 * (seed - 5))   # seed 5 is the published draw
    X = X0.copy()
    D, ok = D0[take], ok0
    alphas = I.alpha_schedule(na)
    hist = []
    for it in range(na):
        r = (D - sel_obs[:, None]) / sel_sd[:, None]
        hist.append(float((r[:, ok] ** 2).mean()))
        X = I.esmda_update(X, D, sel_obs, sel_sd, float(alphas[it]), rho, rng, ok, pr=pr, rtps=rtps)
        Dfull, ok = I.run_ensemble(X, root, C.EST, mask_e, geom, workers=workers)
        D = Dfull[take]
    r = (D - sel_obs[:, None]) / sel_sd[:, None]
    hist.append(float((r[:, ok] ** 2).mean()))
    return X, ok, hist, D


def score(X, ok, q_true):
    ens = np.array([E.q_annual(X[:, i]) for i in np.nonzero(ok)[0]])
    out = MT.point_scores(ens.mean(axis=0), q_true)
    out.update(MT.coverage(ens, q_true))
    out["crps_mcm"] = MT.crps(ens, q_true)
    return out, ens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ne", type=int, default=250)
    ap.add_argument("--na", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rtps", type=float, default=0.7)
    ap.add_argument("--rows", type=str, default="ET,A,B,C,D,E,F,G,H")
    ap.add_argument("--meters", type=str, default="")
    ap.add_argument("--reuse-budget", action="store_true")
    ap.add_argument("--seed", type=int, default=5,
                    help="prior ensemble draw and ES-MDA perturbation seed")
    ap.add_argument("--truth", type=str, default="truth.npz")
    ap.add_argument("--out", type=str, default="ablation.json")
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for posterior_*.npz files and the scratch run root")
    args = ap.parse_args()

    if args.meters:
        ROW_METERS["HM3"] = [int(v) for v in args.meters.split(",")]
        ROW_METERS["HM1"] = ROW_METERS["HM3"][:1]

    tr, mask_t, mask_e, geom = load_setup(args.truth)
    q_true = tr["q_annual"]
    pr = E.prior(mask_e, C.EST)
    X0 = E.sample_prior(pr, args.ne, seed=args.seed)
    root = ROOT / "runs" / ("ens" + args.tag)

    print(f"L0 ablation: ne={args.ne}, na={args.na}, npar={E.NPAR}, "
          f"seed={args.seed}, truth={args.truth}")
    t0 = time.time()
    D0, ok0 = I.run_ensemble(X0, root, C.EST, mask_e, geom, workers=args.workers)
    print(f"prior ensemble {time.time()-t0:.0f}s, {int(ok0.sum())}/{args.ne} converged")

    # Observations are weighted at their instrument error. The structural error of
    # the coarse forward model is estimated after the fact from the residual of the
    # full-information row and reported alongside; on every leg it comes out below the
    # instrument error, so no inflation is applied. See DECISION_LOG.md for the two
    # earlier weightings that did not survive.
    obs_full = np.concatenate([tr["obs_" + k] for k in I.LEGS])
    sd_full = np.concatenate([tr["sig_" + k] for k in I.LEGS])
    budget = None

    out_path = ROOT / "results" / args.out
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    prior_ens = np.array([E.q_annual(X0[:, i]) for i in np.nonzero(ok0)[0]])
    ps, _ = score(X0, ok0, q_true)
    res["PRIOR"] = {"legs": [], "label": "prior, no data", **ps}

    e_star = MT.oracle_efficiency(tr["obs_et"], q_true)
    res["BASELINE"] = {"legs": ["et"], "label": "open loop, efficiency fixed at 0.80",
                       **MT.point_scores(MT.baseline_open_loop(tr["obs_et"]), q_true)}
    res["BASELINE_ORACLE"] = {
        "legs": ["et"], "label": f"open loop, efficiency fitted to truth at {e_star:.3f}",
        **MT.point_scores(MT.baseline_open_loop(tr["obs_et"], e_star), q_true)}

    for key in args.rows.split(","):
        legs = ROWS[key]
        t = time.time()
        X, ok, hist, Drow = run_row(legs, X0, D0, ok0, tr, mask_e, geom,
                                    args.na, args.workers, root, pr, sd_full,
                                    meters=ROW_METERS.get(key),
                                    wells=ROW_WELLS.get(key), rtps=args.rtps,
                                    seed=args.seed)
        if key == "H" and not args.tag:
            idxH = leg_index(tr)
            takeH = np.concatenate([np.arange(idxH[k].start, idxH[k].stop) for k in legs])
            _, budget = I.total_error(Drow[:, ok].mean(axis=1), obs_full[takeH],
                                      sd_full[takeH], geom, legs=legs)
            res["_error_budget"] = budget
            (ROOT / "results" / "error_budget.json").write_text(json.dumps(budget, indent=2))
            np.save(ROOT / "results" / "error_budget.npy", sd_full)
        sc, ens = score(X, ok, q_true)
        eta_hat = X[E.LAYOUT["eta"]][:, ok].mean(axis=1)
        pre_hat = X[E.LAYOUT["preplant"]][:, ok].mean(axis=1)
        eta_true = tr["eta"] if "eta" in tr else C.DIST_ETA
        pre_true = tr["preplant"] if "preplant" in tr else C.DIST_PREPLANT
        pcs_hat = float((10.0 ** X[E.LAYOUT["log_pcs"]][:, ok]).mean())
        rd = MT.resolved_directions(ens, prior_ens)
        res[key] = {
            "legs": list(legs), "label": ROW_LABEL[key], **sc,
            "phi_history": hist,
            "eta_mae": float(np.abs(eta_hat - eta_true).mean()),
            "preplant_mae": float(np.abs(pre_hat - pre_true).mean()),
            "pcs_offset_hat": pcs_hat,
            "pcs_offset_err": pcs_hat - 12.0,
            "n_resolved_90": rd["n_resolved_90"], "n_unresolved": rd["n_unresolved"],
            "effective_dim": rd["effective_dim"],
            "seconds": time.time() - t,
        }
        np.savez_compressed(ROOT / "results" / f"posterior_{key}{args.tag}.npz",
                            X=X, ok=ok, ens=ens, variance_ratio=rd["variance_ratio"])
        print(f"  {key:3s} {ROW_LABEL[key]:38s} MAE {sc['mae_mcm']:7.2f} Mm3/yr  "
              f"MAPE {sc['mape_pct']:5.1f}%  cover90 {sc['cover_90']:.2f}  "
              f"({time.time()-t:.0f}s)")

    out_path.write_text(json.dumps(res, indent=2))
    print("\nwrote results/ablation.json")
    print(f"{'row':4s} {'observations':40s} {'MAE':>9s} {'MAPE':>7s} {'bias':>9s} "
          f"{'cov90':>6s} {'CRPS':>7s}")
    for k, v in res.items():
        if k.startswith("_"):
            continue
        print(f"{k:4s} {v['label']:40s} {v['mae_mcm']:9.2f} {v['mape_pct']:6.1f}% "
              f"{v['bias_mcm']:9.2f} {v.get('cover_90', float('nan')):6.2f} "
              f"{v.get('crps_mcm', float('nan')):7.2f}")


if __name__ == "__main__":
    main()
