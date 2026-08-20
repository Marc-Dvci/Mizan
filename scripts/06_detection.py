"""Detecting abstraction that no evapotranspiration product can see.

An unlicensed well field, an industrial abstraction, a municipal wellhead: none of it
grows a canopy, so none of it appears in a satellite evapotranspiration retrieval, and
an open-loop account of the basin is blind to all of it by construction.

The aquifer is not blind to it. The water still leaves storage, the head still falls,
and the skeleton still compacts. This script plants such a withdrawal in one district,
runs the closure with the evapotranspiration retrieval unchanged, and reports the
statistic a regulator would act on: how far each district's closure estimate departs
from what its own consumptive use can explain.

Usage:
  python scripts/00_truth.py --hidden 3:40 --out truth_hidden.npz
  python scripts/06_detection.py [--ne 250] [--na 8]
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
sys.path.insert(0, str(ROOT / "scripts"))

from importlib import import_module

from mizan import config as C, estimator as E, fields, forcing as F
from mizan import inversion as I, metrics as MT, observations as O

AB = import_module("01_ablation")


def ratio_statistic(q_hat: np.ndarray, obs_et: np.ndarray) -> np.ndarray:
    """Per district: closure estimate divided by what consumptive use can explain.

    The denominator is the open-loop account at its published fixed efficiency, which
    is the number a regulator already has. A district whose books balance sits with
    every other district; a district drawing water that grows no canopy does not.
    """
    return q_hat.sum(axis=1) / MT.baseline_open_loop(obs_et).sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ne", type=int, default=250)
    ap.add_argument("--na", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rtps", type=float, default=0.7)
    ap.add_argument("--truth", type=str, default="truth_hidden.npz")
    args = ap.parse_args()

    tr = np.load(ROOT / "results" / args.truth)
    d_hidden = int(tr["hidden_district"])
    q_true, q_vis = tr["q_annual"], tr["q_visible"]
    hidden = (q_true - q_vis).sum(axis=1)
    print(f"planted withdrawal: district {d_hidden}, "
          f"{hidden[d_hidden]/1e6/C.NYEAR:.0f} Mm3/yr with no canopy, "
          f"{hidden[d_hidden]/q_true[d_hidden].sum()*100:.0f}% of that district's total")

    mask_t = F.pivot_mask(C.TRUTH)
    mask_e = fields.upscale_mask(mask_t, int(C.EST.delr_m // C.TRUTH.delr_m))
    geom = O.Geometry(well_xy=tr["well_xy"], well_seen=tr["well_seen"],
                      insar_xy=tr["insar_xy"], insar_ref_xy=tr["insar_ref_xy"],
                      insar_epochs=np.arange(C.INSAR_STACK_MONTHS - 1, C.NPER,
                                             C.INSAR_STACK_MONTHS))
    pr = E.prior(mask_e, C.EST)
    X0 = E.sample_prior(pr, args.ne)
    root = ROOT / "runs" / "det"

    sd_full = np.concatenate([tr["sig_" + k] for k in I.LEGS])
    t0 = time.time()
    D0, ok0 = I.run_ensemble(X0, root, C.EST, mask_e, geom, workers=args.workers)
    print(f"prior ensemble {time.time()-t0:.0f}s, {int(ok0.sum())}/{args.ne} converged")

    legs = tuple(k for k in I.LEGS if k != "meter")
    X, ok, hist, _ = AB.run_row(legs, X0, D0, ok0, tr, mask_e, geom, args.na,
                                args.workers, root, pr, sd_full, rtps=args.rtps)
    ens = np.array([E.q_annual(X[:, i]) for i in np.nonzero(ok)[0]])
    q_hat = ens.mean(axis=0)

    r = ratio_statistic(q_hat, tr["obs_et"])
    others = np.array([r[d] for d in range(C.NDIST) if d != d_hidden])
    z = (r[d_hidden] - others.mean()) / others.std(ddof=1)

    sc = MT.point_scores(q_hat, q_true)
    sc.update(MT.coverage(ens, q_true))
    ol = MT.baseline_open_loop(tr["obs_et"])
    sc_ol = MT.point_scores(ol, q_true)

    print(f"\nclosure MAE {sc['mae_mcm']:.2f} Mm3/yr, open loop {sc_ol['mae_mcm']:.2f}")
    print(f"\n{'district':9s} {'closure':>10s} {'open loop':>10s} {'ratio':>7s} "
          f"{'truth':>10s} {'recovered':>10s}")
    for d in range(C.NDIST):
        mark = "  <- planted" if d == d_hidden else ""
        print(f"D{d:<8d} {q_hat[d].sum()/1e6:10.0f} {ol[d].sum()/1e6:10.0f} "
              f"{r[d]:7.2f} {q_true[d].sum()/1e6:10.0f} "
              f"{q_hat[d].sum()/q_true[d].sum()*100:9.0f}%{mark}")
    print(f"\nthe planted district sits {z:.1f} standard deviations above the others on "
          f"the ratio statistic")
    print(f"the open-loop account attributes {ol[d_hidden].sum()/q_true[d_hidden].sum()*100:.0f}% "
          f"of that district's true abstraction")

    res = {
        "hidden_district": d_hidden,
        "hidden_mcm_per_year": float(hidden[d_hidden] / 1e6 / C.NYEAR),
        "hidden_share_of_district": float(hidden[d_hidden] / q_true[d_hidden].sum()),
        "ratio": {f"D{d}": float(r[d]) for d in range(C.NDIST)},
        "z_score": float(z),
        "closure": sc, "open_loop": sc_ol,
        "closure_recovery_pct": float(q_hat[d_hidden].sum() / q_true[d_hidden].sum() * 100),
        "open_loop_recovery_pct": float(ol[d_hidden].sum() / q_true[d_hidden].sum() * 100),
        "phi_history": hist,
    }
    (ROOT / "results" / "detection.json").write_text(json.dumps(res, indent=2))
    np.savez_compressed(ROOT / "results" / "detection.npz", X=X, ok=ok, ens=ens, ratio=r)
    print("\nwrote results/detection.json")


if __name__ == "__main__":
    main()
