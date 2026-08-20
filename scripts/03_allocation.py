"""Workstream C: what the posterior lets a regulator decide.

Two questions, in the order they matter for a fossil aquifer.

**How much can be taken.** Permanent storage loss against delivered water over a
twenty-year horizon, computed in full MODFLOW across the posterior ensemble, so the
frontier carries the uncertainty of the abstraction estimate rather than assuming it
away. This is the curve that prices the irreversible column.

**How it should be spread.** At a fixed delivered volume, the quota vector that
minimises the tail of permanent loss, against the uniform proportional cut regulators
actually apply. Both are re-run in full MODFLOW with the inelastic switch active and
the surrogate-to-simulator discrepancy is reported.

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
    """Permanent storage loss added over the horizon, m3, in full MODFLOW."""
    i, x, q = arg
    try:
        p = E.to_params(x, C.EST)
        p.q_monthly = AL.build_future_q(E.q_annual(x), q, x[E.LAYOUT["preplant"]])
        M.build(_W["ws"], C.EST, p, _W["mask"], inelastic=True)
        if not M.run(_W["ws"]):
            return i, None
        inel = O.read_inelastic(_W["ws"], C.EST)
        a = C.EST.delr_m ** 2
        return i, float((inel[-1].sum() - inel[C.NPER - 1].sum()) * a)
    except Exception:
        return i, None


def pmap(fn, items, workers, mask, qref):
    out = [None] * len(items)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(mask, qref)) as ex:
        for i, r in ex.map(fn, items, chunksize=1):
            out[i] = r
    return out


def band(v):
    v = np.asarray([x for x in v if x is not None], dtype=float)
    return {
        "mean_mcm": float(v.mean() / 1e6),
        "p10_mcm": float(np.quantile(v, 0.10) / 1e6),
        "p90_mcm": float(np.quantile(v, 0.90) / 1e6),
        "cvar90_mcm": float(np.sort(v)[int(0.9 * len(v)):].mean() / 1e6),
        "n": int(v.size),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=24)
    ap.add_argument("--cut", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--floor", type=float, default=0.25)
    args = ap.parse_args()

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
        b = band(vals)
        b.update({"cut": cut, "delivered_km3": float(q.sum() / 1e9)})
        frontier.append(b)
        print(f"{q.sum()/1e9:10.2f} {cut*100:6.0f} | {b['mean_mcm']:10.0f} "
              f"{b['p10_mcm']:10.0f} {b['p90_mcm']:10.0f}")
    res["frontier"] = frontier

    # -- how it should be spread, at one operating point --------------------------
    total = (1.0 - args.cut) * bau
    q_uni = AL.uniform_policy(qhat.mean(axis=0), total)
    cap = np.tile(1.20 * qhat.mean(axis=0).max(axis=1)[:, None], (1, AL.HORIZON_Y))
    floor = args.floor * q_uni.sum(axis=1)

    q_opt, status = AL.optimise(surrs, q_ref, total, cap, floor, beta=0.90)
    q_cc, status_cc = AL.optimise(surrs, q_ref, total, cap, floor, beta=0.90, chance=0.95)
    print("")
    print(f"at a {args.cut*100:.0f}% cut, floor {args.floor:.2f} of the uniform share")
    print(f"  allocation status {status}; chance-constrained {status_cc}")

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
        res[name]["simulator"] = band(vals)
        d = res[name]["simulator"]["cvar90_mcm"]
        print(f"  {name:20s} surrogate CVaR90 {res[name]['loss_cvar90_mcm']:8.0f} Mm3   "
              f"simulator {d:8.0f} Mm3   discrepancy "
              f"{(res[name]['loss_cvar90_mcm']-d)/d*100:+.0f}%")

    a = res["uniform"]["simulator"]["cvar90_mcm"]
    b = res.get("risk_bounded", res["uniform"])["simulator"]["cvar90_mcm"]
    res["reallocation_gain_pct"] = float((a - b) / a * 100.0)
    print("")
    print(f"reallocation at equal delivered water changes permanent loss at the 90 per "
          f"cent tail by {res['reallocation_gain_pct']:+.1f}%")

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
    (ROOT / "results" / "allocation.json").write_text(json.dumps(res, indent=2))
    print("wrote results/allocation.json")


if __name__ == "__main__":
    main()
