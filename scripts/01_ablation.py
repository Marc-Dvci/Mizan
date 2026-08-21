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


def run_row(legs, X0, D0, ok0, tr, mask_e, geom, na, workers, root, pr,
            obs_full, sd_full, meters=None, wells=None, rtps=0.7, seed=5):
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


def gain_posterior(X, ok, pr):
    """What the data did to the mascon gain, and what the gain does to the answer.

    Two numbers a juror asks for and the ablation grid never reported. The first is
    whether the observations identify the gain at all, read as the share of its prior
    variance the posterior removes. The second is how much of the absolute scale rides on
    it, read as the covariance between the gain and the basin total across the ensemble.
    That second one is a posterior sensitivity rather than a controlled sweep, because
    every other parameter is free to move along the gain axis, so it is reported under a
    name that says so.
    """
    ia = E.LAYOUT["grace_alpha"]
    a = np.asarray(X[ia][:, ok]).ravel()
    sd0 = float(np.asarray(pr.sd[ia]).ravel()[0])
    Q = (10.0 ** X[E.LAYOUT["logq"]][:, ok]).sum(axis=0).ravel() / 1e6
    out = {"alpha_hat": float(a.mean()), "alpha_sd_post": float(a.std()),
           "alpha_prior_sd": sd0,
           "alpha_var_removed": float(1.0 - (a.std() / sd0) ** 2) if sd0 > 0 else 0.0,
           "basin_total_mcm": float(Q.mean())}
    if a.std() > 1e-9 and Q.std() > 0:
        slope = float(np.polyfit(a, Q, 1)[0])
        out["dQ_dalpha_mcm"] = slope
        out["alpha_Q_corr"] = float(np.corrcoef(a, Q)[0, 1])
        out["scale_sensitivity_pct_per_prior_sd"] = float(
            100.0 * slope * sd0 / max(Q.mean(), 1e-9))
    return out


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
    # The mascon gain multiplies the gravity leg, and basin storage falls close to
    # linearly, so the gain and a free external trend absorb the absolute scale of the
    # answer between them. The entry treats the gain as computed rather than fitted,
    # which is a prior. These two arguments are how that choice is put on an axis and
    # scored, instead of being asserted.
    ap.add_argument("--alpha-sd", type=float, default=None,
                    help="override the mascon gain prior standard deviation; the shipped "
                         "configuration is 0.04")
    ap.add_argument("--alpha-mean", type=float, default=None,
                    help="override the mascon gain prior mean; the shipped value is 0.85")
    # The L0 twin generates its gravity leg at C.GRACE_SIGMA_MM. The uncertainty the
    # mascon product publishes over the Saq is larger than that, and script 24 measures
    # it, so the row can be repeated with the leg degraded to the published figure.
    ap.add_argument("--grace-sigma", type=float, default=None,
                    help="degrade the gravity leg to this instrument error in mm, "
                         "adding the variance difference to the observation and telling "
                         "the estimator about it; the twin generates at "
                         f"{C.GRACE_SIGMA_MM:.0f}")
    # The external mass trend is constrained to plus or minus 1.0 mm/yr on the physical
    # argument that in a hyper-arid basin the trend in total water storage is the trend
    # in groundwater. On the target basin the L3 controls measure the regional trend over
    # unirrigated desert at several times that, so the width of this prior is an axis in
    # its own right and is scored rather than argued.
    ap.add_argument("--drift-sd", type=float, default=None,
                    help="override the external mass trend prior standard deviation in "
                         "mm/yr; the shipped configuration is 1.0")
    ap.add_argument("--drift-free", action="store_true",
                    help="release the external mass trend from its plus or minus 1.0 "
                         "mm/yr constraint, which is the configuration the decision log "
                         "records as rejected")
    args = ap.parse_args()

    if args.meters:
        ROW_METERS["HM3"] = [int(v) for v in args.meters.split(",")]
        ROW_METERS["HM1"] = ROW_METERS["HM3"][:1]

    tr, mask_t, mask_e, geom = load_setup(args.truth)
    q_true = tr["q_annual"]
    pr = E.prior(mask_e, C.EST)
    ia = E.LAYOUT["grace_alpha"]
    if args.alpha_mean is not None:
        pr.mean[ia] = args.alpha_mean
    if args.alpha_sd is not None:
        pr.sd[ia] = args.alpha_sd
    if args.drift_free:
        pr.sd[E.LAYOUT["grace_drift"]] = np.array([3.0, 4.0, 4.0])
    if args.drift_sd is not None:
        pr.sd[E.LAYOUT["grace_drift"]] = np.array([args.drift_sd, 4.0, 4.0])
    gain_cfg = {"alpha_mean": float(np.asarray(pr.mean[ia]).ravel()[0]),
                "alpha_sd": float(np.asarray(pr.sd[ia]).ravel()[0]),
                "drift_trend_sd": float(np.asarray(pr.sd[E.LAYOUT["grace_drift"]]).ravel()[0]),
                "grace_sigma_mm": float(args.grace_sigma if args.grace_sigma is not None
                                        else C.GRACE_SIGMA_MM)}
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
    obs = {k: np.array(tr["obs_" + k], float) for k in I.LEGS}
    sig = {k: np.array(tr["sig_" + k], float) for k in I.LEGS}
    if args.grace_sigma is not None:
        # The leg is already carrying the twin's own noise, so only the variance
        # difference is added. Degrading an observation the estimator still trusts at the
        # old error would be a different and dishonest experiment, so the reported error
        # moves with it.
        s0 = float(C.GRACE_SIGMA_MM)
        extra = float(args.grace_sigma) ** 2 - s0 ** 2
        if extra < 0:
            raise SystemExit(f"--grace-sigma below the twin's own {s0:.1f} mm cannot be "
                             "reached by adding noise; regenerate the truth instead")
        rng = np.random.default_rng(args.seed + 9001)
        obs["grace"] = obs["grace"] + np.sqrt(extra) * rng.standard_normal(
            obs["grace"].shape)
        sig["grace"] = np.full(sig["grace"].shape, float(args.grace_sigma))
        print(f"gravity leg degraded from {s0:.1f} to {args.grace_sigma:.1f} mm")
    obs_full = np.concatenate([obs[k] for k in I.LEGS])
    sd_full = np.concatenate([sig[k] for k in I.LEGS])
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
                                    args.na, args.workers, root, pr,
                                    obs_full, sd_full,
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
            **gain_posterior(X, ok, pr),
        }
        np.savez_compressed(ROOT / "results" / f"posterior_{key}{args.tag}.npz",
                            X=X, ok=ok, ens=ens, variance_ratio=rd["variance_ratio"])
        print(f"  {key:3s} {ROW_LABEL[key]:38s} MAE {sc['mae_mcm']:7.2f} Mm3/yr  "
              f"MAPE {sc['mape_pct']:5.1f}%  cover90 {sc['cover_90']:.2f}  "
              f"({time.time()-t:.0f}s)")

    res["_meta"] = {"seed": args.seed, "ne": args.ne, "na": args.na,
                    "truth": args.truth, "rtps": args.rtps, **gain_cfg}
    out_path.write_text(json.dumps(res, indent=2))
    print(f"\nwrote results/{args.out}")
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
