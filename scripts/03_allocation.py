"""Workstream C: what the posterior lets a regulator decide.

Two questions, in the order they matter for a fossil aquifer.

**How much can be taken.** Permanent storage loss against delivered water over a
twenty-year horizon, computed in full MODFLOW across the posterior ensemble, so the
frontier carries the uncertainty of the abstraction estimate rather than assuming it
away. This is the curve that prices the irreversible column.

**Experimental spatial allocation.** At a fixed delivered volume, a response-matrix
programme tests quota vectors against the uniform proportional cut. These diagnostics
are retained for method development and are not part of the submitted decision claim.

Usage:  python scripts/03_allocation.py [--members 24] [--cut 0.15]
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

from mizan import allocation as AL, config as C, estimator as E, fields, forcing as F
from mizan import model as M, observations as O

RUNS = ROOT / "runs" / "alloc"
CUTS = (0.00, 0.10, 0.20, 0.30, 0.40, 0.50)
_W: dict = {}


def _init(mask, qref):
    _W["mask"] = mask
    _W["qref"] = qref
    _W["ws"] = RUNS / f"w{os.getpid()}"


def _surr(arg):
    i, x = arg
    try:
        return i, AL.member_surrogate(x, _W["ws"], C.EST, _W["mask"], _W["qref"])
    except Exception:
        return i, None


def _lti(arg):
    i, x, surr = arg
    try:
        return i, AL.lti_check(x, _W["ws"], C.EST, _W["mask"], _W["qref"], surr)
    except Exception:
        return i, None


def _verify(arg):
    """Permanent loss and threshold exceedance in full MODFLOW."""
    i, x, q = arg
    try:
        p = E.to_params(x, C.EST)
        p.q_monthly = AL.build_future_q(E.q_annual(x), q, x[E.LAYOUT["preplant"]])
        M.build(_W["ws"], C.EST, p, _W["mask"], inelastic=True)
        if not M.run(_W["ws"]):
            return i, None
        st = O.read_state(_W["ws"], C.EST, sy=p.sy)
        inel = O.read_inelastic(_W["ws"], C.EST)
        a = C.EST.delr_m ** 2
        loss = float((inel[-1].sum() - inel[C.NPER - 1].sum()) * a)
        hist = st.head[:C.NPER, C.CSUB_LAYER]
        pcs = np.minimum(C.H_INIT - p.pcs_offset, hist.min(axis=0))
        future = st.head[C.NPER:, C.CSUB_LAYER]
        worst = float(np.max(pcs[None, :, :] - future))
        return i, {"loss_m3": loss, "worst_exceed_m": worst}
    except Exception:
        return i, None


def pmap(fn, items, workers, mask, qref):
    out = [None] * len(items)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(mask, qref)) as ex:
        for i, r in ex.map(fn, items, chunksize=1):
            out[i] = r
    return out


def verification_summary(values):
    """Exact empirical summaries of completed full-simulator runs."""
    valid = [x for x in values if x is not None]
    loss = np.asarray([x["loss_m3"] for x in valid], dtype=float)
    worst = np.asarray([x["worst_exceed_m"] for x in valid], dtype=float)
    if not len(valid):
        raise RuntimeError("all full-simulator verification runs failed")
    return {
        "mean_mcm": float(loss.mean() / 1e6),
        "p10_mcm": float(np.quantile(loss, 0.10) / 1e6),
        "p90_mcm": float(np.quantile(loss, 0.90) / 1e6),
        "cvar90_mcm": float(AL.empirical_cvar(loss, 0.90) / 1e6),
        "p_exceed": float((worst > 1.0e-6).mean()),
        "exceed_cvar95_m": float(AL.empirical_cvar(worst, 0.95)),
        "worst_exceed_m": float(worst.max()),
        "samples_loss_m3": loss.tolist(),
        "samples_worst_exceed_m": worst.tolist(),
        "n": int(loss.size),
    }


def verify_saved_interpolations(args):
    """Full-model safety scan from the uniform policy toward a saved surrogate optimum."""
    saved = np.load(ROOT / "results" / "allocation.npz")
    post = np.load(ROOT / "results" / "posterior_H.npz")
    X = post["X"]
    training_members = saved["members"].astype(int)
    take = training_members
    cohort = "training"
    if args.holdout_members:
        ok = np.nonzero(post["ok"])[0]
        available = np.setdiff1d(ok, training_members)
        if args.holdout_members > len(available):
            raise ValueError("requested more holdout members than are available")
        take = available[np.linspace(0, len(available) - 1,
                                     args.holdout_members).astype(int)]
        cohort = "holdout"
    q_ref = saved["q_ref"]
    q_uniform = saved["q_uniform"]
    q_opt = saved["q_opt"]
    if q_opt.shape != q_uniform.shape:
        raise RuntimeError("results/allocation.npz has no valid experimental optimum")
    mask_e = fields.upscale_mask(F.pivot_mask(C.TRUTH), 2)
    fractions = tuple(float(x) for x in args.interpolation_fractions.split(","))
    if any(not 0.0 <= x <= 1.0 for x in fractions):
        raise ValueError("interpolation fractions must be between zero and one")

    out = {"cohort": cohort, "members": take.tolist(), "policies": []}
    for fraction in fractions:
        q = q_uniform + fraction * (q_opt - q_uniform)
        vals = pmap(_verify, [(i, X[:, j], q) for i, j in enumerate(take)],
                    args.workers, mask_e, q_ref)
        summary = verification_summary(vals)
        summary.update({"fraction_toward_surrogate_optimum": fraction,
                        "delivered_km3": float(q.sum() / 1e9)})
        out["policies"].append(summary)
        print(f"fraction {fraction:5.2f}: CVaR90 {summary['cvar90_mcm']:8.1f} Mm3, "
              f"P(exceed) {summary['p_exceed']:.3f}, "
              f"exceed-CVaR95 {summary['exceed_cvar95_m']:.2f} m")
    suffix = "_holdout" if args.holdout_members else ""
    path = ROOT / "results" / f"allocation_interpolation{suffix}.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=24)
    ap.add_argument("--cut", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--floor", type=float, default=0.25)
    ap.add_argument("--verify-saved-interpolations", action="store_true")
    ap.add_argument("--interpolation-fractions", default="0.10,0.25,0.50,0.75")
    ap.add_argument("--holdout-members", type=int, default=0)
    args = ap.parse_args()

    if args.verify_saved_interpolations:
        verify_saved_interpolations(args)
        return

    post = np.load(ROOT / "results" / "posterior_H.npz")
    X, ok = post["X"], post["ok"]
    idx = np.nonzero(ok)[0]
    take = idx[np.linspace(0, len(idx) - 1, args.members).astype(int)]
    mask_e = fields.upscale_mask(F.pivot_mask(C.TRUTH), 2)

    qhat = np.array([E.q_annual(X[:, i]) for i in take])
    q_last = qhat[:, :, -1].mean(axis=0)
    q_ref = np.tile(q_last[:, None], (1, AL.HORIZON_Y))
    bau = q_ref.sum()
    print(f"business as usual {bau/1e9:.2f} km3 over {AL.HORIZON_Y} yr, "
          f"{len(take)} posterior members")

    t0 = time.time()
    surrs = pmap(_surr, [(i, X[:, j]) for i, j in enumerate(take)],
                 args.workers, mask_e, q_ref)
    keep = [i for i, s in enumerate(surrs) if s is not None]
    surrs = [surrs[i] for i in keep]
    take = take[keep]
    print(f"{len(surrs)} member surrogates in {time.time()-t0:.0f}s, "
          f"{surrs[0].R.shape[0]} drawdown zones")

    hist_loss = np.array([s.loss_hist for s in surrs])
    print(f"permanent loss already taken over the record: "
          f"{hist_loss.mean()/1e6:.0f} Mm3 (posterior mean)")

    nl = min(3, len(surrs))
    lti = [v for v in pmap(_lti, [(i, X[:, take[i]], surrs[i]) for i in range(nl)],
                           args.workers, mask_e, q_ref) if v is not None]
    lti_max = float(np.mean([v["max_rel_error"] for v in lti])) if lti else float("nan")
    print(f"time invariance of the response matrix, {len(lti)} members: mean worst-case "
          f"error {lti_max*100:.1f}% of the pulse response")

    res = {"members": len(surrs), "bau_km3": bau / 1e9,
           "lti_mean_max_rel_error": lti_max,
           "hist_permanent_loss_mcm": float(hist_loss.mean() / 1e6),
           "zones": int(surrs[0].R.shape[0])}

    # -- the frontier: how much can be taken -------------------------------------
    print("")
    print(f"{'delivered':>10s} {'cut':>6s} | permanent loss over horizon, Mm3")
    print(f"{'km3':>10s} {'%':>6s} | {'mean':>10s} {'p10':>10s} {'p90':>10s}")
    frontier = []
    for cut in CUTS:
        q = AL.uniform_policy(qhat.mean(axis=0), (1.0 - cut) * bau)
        vals = pmap(_verify, [(i, X[:, j], q) for i, j in enumerate(take)],
                    args.workers, mask_e, q_ref)
        b = verification_summary(vals)
        b.update({"cut": cut, "delivered_km3": float(q.sum() / 1e9)})
        frontier.append(b)
        print(f"{q.sum()/1e9:10.2f} {cut*100:6.0f} | {b['mean_mcm']:10.0f} "
              f"{b['p10_mcm']:10.0f} {b['p90_mcm']:10.0f}")
    res["frontier"] = frontier

    # -- experimental spatial allocation at one operating point -------------------
    total = (1.0 - args.cut) * bau
    q_uni = AL.uniform_policy(qhat.mean(axis=0), total)
    cap = np.tile(1.20 * qhat.mean(axis=0).max(axis=1)[:, None], (1, AL.HORIZON_Y))
    floor = args.floor * q_uni.sum(axis=1)

    opt = AL.optimise(surrs, q_ref, total, cap, floor, beta=0.90)
    opt_cc = AL.optimise(surrs, q_ref, total, cap, floor, beta=0.90, chance=0.95)
    q_opt, q_cc = opt.policy, opt_cc.policy
    print("")
    print(f"at a {args.cut*100:.0f}% cut, floor {args.floor:.2f} of the uniform share")
    print(f"  experimental allocation {opt.solver}/{opt.status}, "
          f"post-check feasible={opt.feasible}")
    print(f"  experimental chance constraint {opt_cc.solver}/{opt_cc.status}, "
          f"post-check feasible={opt_cc.feasible}")
    res["experimental_optimisation"] = {
        "risk_bounded": opt.summary(),
        "chance_constrained": opt_cc.summary(),
        "floor_fraction_of_uniform_district_total": float(args.floor),
        "district_year_cap_fraction_of_historical_peak": 1.20,
    }

    pol = {"uniform": q_uni}
    if q_opt is not None:
        pol["risk_bounded"] = q_opt
    if q_cc is not None:
        pol["chance_constrained"] = q_cc

    for name, q in pol.items():
        res[name] = AL.evaluate_policy(surrs, q, q_ref)
        res[name]["delivered_km3"] = float(q.sum() / 1e9)
        vals = pmap(_verify, [(i, X[:, j], q) for i, j in enumerate(take)],
                    args.workers, mask_e, q_ref)
        res[name]["simulator"] = verification_summary(vals)
        d = res[name]["simulator"]["cvar90_mcm"]
        print(f"  {name:20s} surrogate CVaR90 {res[name]['loss_cvar90_mcm']:8.0f} Mm3   "
              f"simulator {d:8.0f} Mm3   discrepancy "
              f"{(res[name]['loss_cvar90_mcm']-d)/d*100:+.0f}%")

    if "risk_bounded" in res:
        a = res["uniform"]["simulator"]["cvar90_mcm"]
        b = res["risk_bounded"]["simulator"]["cvar90_mcm"]
        res["experimental_reallocation_gain_pct"] = float((a - b) / a * 100.0)
        print("")
        print(f"experimental reallocation changes permanent loss at the 90 per cent tail "
              f"by {res['experimental_reallocation_gain_pct']:+.1f}%")
    else:
        res["experimental_reallocation_gain_pct"] = None

    f0, f1 = frontier[0], frontier[-1]
    dq = f0["delivered_km3"] - f1["delivered_km3"]
    dl = f0["mean_mcm"] - f1["mean_mcm"]
    res["marginal_capacity_per_km3"] = float(dl / dq) if dq else float("nan")
    print(f"cutting delivery by {dq:.2f} km3 over the horizon avoids {dl:.0f} Mm3 of "
          f"permanent capacity loss, {dl/dq:.0f} Mm3 per km3 not taken")

    np.savez_compressed(ROOT / "results" / "allocation.npz",
                        q_ref=q_ref, q_uniform=q_uni,
                        q_opt=q_opt if q_opt is not None else np.zeros(0),
                        q_cc=q_cc if q_cc is not None else np.zeros(0),
                        members=take)
    (ROOT / "results" / "allocation.json").write_text(
        json.dumps(res, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("wrote results/allocation.json")


if __name__ == "__main__":
    main()
