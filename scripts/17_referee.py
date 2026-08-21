"""L2b: can the aquifer referee a water account that was built without it?

Every published estimate of irrigation abstraction is an account of the land surface.
Mapped area times a published depth is one. That depth corrected for the year's
precipitation deficit is another. Evapotranspiration over a fixed efficiency is a third.
They disagree, none of them carries an uncertainty, and in an unmetered basin there is
nothing to choose between them.

The aquifer is the one witness that is not part of that argument. Water that is pumped
leaves storage, and storage is what a water level measures. So the question this script
asks is not "what is the abstraction" but

    given this account of abstraction, is there any physically admissible aquifer that
    reproduces the observed water levels?

For each candidate account the abstraction is held fixed and only the aquifer's own
properties are estimated against the head record: storage, the thickness multiplier, the
mean recharge, the boundary conductance and the conductivity field. What is reported is
the head misfit the best admissible aquifer can reach. An account that asks the aquifer
for water it cannot have supplied is not rescued by any parameter draw, and its misfit
stays high.

The test never sees a meter. The meters are opened once, at the end, to ask whether the
ranking the aquifer produced is the ranking the meters would have produced.

    python scripts/17_referee.py --ne 100 --na 3 --tag _v4

Writes results/referee{tag}.json.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mizan import inversion as I, ks_data as K, ks_run as R, metrics as MT

RES = ROOT / "results"


def candidates(a: dict, closure: np.ndarray, q_true: np.ndarray) -> dict:
    """The accounts to be refereed, m3/yr, shape (6, 25) each.

    Five are accounts a practitioner could write down today, and none of the five uses a
    meter. Two more are the same flat account rescaled by a factor the record cannot
    support, and they are here so the test is shown to fail when it should: an
    instrument that ranks everything the same way ranks nothing.
    """
    area = a["irr_area"]
    P = K.precipitation()
    deficit = P.mean(axis=1, keepdims=True) - P              # mm, positive when dry
    out = {
        "FLAT": ("mapped irrigated area x one acre-foot per acre",
                 area * R.PRIOR_DEPTH_M),
        "WATERBAL": ("the same, plus half the year's precipitation deficit",
                     area * (R.PRIOR_DEPTH_M + 0.5 * deficit / 1000.0)),
        "OPENLOOP": ("unmixed evapotranspiration over a fixed efficiency of 0.80",
                     a["et_obs"] / 0.80),
        "CLOSURE": ("the closure posterior mean", closure),
        "METERED": ("the withheld meters themselves", q_true),
        "FLAT_HIGH": ("the flat account inflated by half", area * R.PRIOR_DEPTH_M * 1.5),
        "FLAT_LOW": ("the flat account cut to three fifths", area * R.PRIOR_DEPTH_M * 0.6),
    }
    return {k: (lab, np.asarray(v, dtype=float)) for k, (lab, v) in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ne", type=int, default=100)
    ap.add_argument("--na", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--tag", type=str, default="_v4")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    drv = importlib.import_module("11_kansas_run")
    a = drv.assemble(pool=True)
    region, ctx = a["region"], a["ctx"]
    n_et = R.NDIST * R.NYEAR

    head_obs = R.head_anomaly(a["wl"], ctx)
    q_true, meta = K.metered_annual()
    post = np.load(RES / ("kansas_posterior_ETH" + args.tag + ".npz"))
    closure = post["ens"].mean(axis=0)

    # The head error is the one the published run uses, so the misfit reported here is
    # on the same scale as the phi the inversion reports.
    kj = json.loads((RES / ("kansas" + args.tag + ".json")).read_text())
    sd_head = float(kj["_error_budget"]["head"]["total"])
    print("head observations {}, error {:.3f} m".format(head_obs.size, sd_head))

    cands = candidates(a, closure, q_true)
    pr = R.prior(region, a["irr_area"])
    root = ROOT / "runs" / ("ref" + args.tag)
    idx = np.arange(n_et, n_et + head_obs.size)
    rho = R.taper(ctx)[:, idx].copy()
    # The abstraction block is frozen: the update may move the aquifer, never the
    # account being tested. Zeroing its localisation row is what freezes it.
    rho[R.LAYOUT["logq"], :] = 0.0
    rho[R.LAYOUT["eta"], :] = 0.0

    out_name = args.out or ("referee" + args.tag + ".json")
    res = {"_meta": {"ne": args.ne, "na": args.na, "seed": args.seed,
                     "tag": args.tag, "sd_head": sd_head,
                     "n_obs_head": int(head_obs.size), **meta}}
    for key, (label, q) in cands.items():
        t0 = time.time()
        logq = np.log10(np.maximum(q, 1.0)).ravel()[:, None]
        X = R.sample_prior(pr, args.ne, seed=args.seed)
        X = R.clip(X, pr)
        X[R.LAYOUT["logq"]] = logq
        rng = np.random.default_rng(args.seed)
        D, ok = R.run_ensemble(X, root, ctx, workers=args.workers)
        hist = []
        alphas = I.alpha_schedule(args.na)
        for it in range(args.na):
            r = (D[idx] - head_obs[:, None]) / sd_head
            hist.append(float((r[:, ok] ** 2).mean()))
            X = I.esmda_update(X, D[idx], head_obs, np.full(head_obs.size, sd_head),
                               float(alphas[it]), rho, rng, ok, pr=pr, rtps=0.7)
            X[R.LAYOUT["logq"]] = logq
            D, ok = R.run_ensemble(X, root, ctx, workers=args.workers)
        r = (D[idx] - head_obs[:, None]) / sd_head
        hist.append(float((r[:, ok] ** 2).mean()))
        rms = float(np.sqrt(((D[idx][:, ok].mean(axis=1) - head_obs) ** 2).mean()))
        sc = MT.point_scores(q, q_true)
        res[key] = {"label": label, "phi_head": hist[-1], "phi_history": hist,
                    "head_rms_m": rms, "converged": int(ok.sum()),
                    "mae_mcm": sc["mae_mcm"], "mape_pct": sc["mape_pct"],
                    "basin_bias_pct": sc["basin_bias_pct"],
                    "seconds": time.time() - t0}
        print("  {:10s} phi_head {:8.3f}  head rms {:5.2f} m  MAE {:6.2f}  "
              "({}/{}, {:.0f}s)".format(key, hist[-1], rms, sc["mae_mcm"],
                                        int(ok.sum()), args.ne, time.time() - t0))
        (RES / out_name).write_text(json.dumps(res, indent=2))

    keys = [k for k in res if not k.startswith("_")]
    phi = np.array([res[k]["phi_head"] for k in keys])
    mae = np.array([res[k]["mae_mcm"] for k in keys])
    order_phi = [keys[i] for i in np.argsort(phi)]
    order_mae = [keys[i] for i in np.argsort(mae)]
    # Spearman on seven points, written out rather than imported, so the number in the
    # results file does not depend on which scipy is installed.
    def rank(v):
        o = np.argsort(np.argsort(v)).astype(float)
        return o
    rp, rm = rank(phi), rank(mae)
    rho_s = float(np.corrcoef(rp, rm)[0, 1])
    res["_ranking"] = {"by_head_misfit": order_phi, "by_metered_error": order_mae,
                       "spearman": rho_s,
                       "metered_rank_by_head": order_phi.index("METERED") + 1,
                       "n_accounts": len(keys)}
    (RES / out_name).write_text(json.dumps(res, indent=2))
    print("\nranked by head misfit, never seeing a meter: " + " < ".join(order_phi))
    print("ranked by the meters:                         " + " < ".join(order_mae))
    print("Spearman {:+.3f}; the metered account ranks {} of {} on head misfit alone"
          .format(rho_s, order_phi.index("METERED") + 1, len(keys)))


if __name__ == "__main__":
    main()
