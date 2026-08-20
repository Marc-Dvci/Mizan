"""How far the coarse forward model can fit the data, and how the estimate moves as it
gets there.

Runs the full four-leg inversion at nominal instrument error for a longer schedule,
recording per-leg residual and abstraction score at every iteration. The converged
residual is what the total error budget has to be built from; a residual taken before
convergence is mostly parameter error and inflates the budget by two orders of
magnitude.

Usage:  python scripts/02_convergence.py [--ne 200] [--na 6]
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mizan import config as C, forcing as F, fields, estimator as E
from mizan import inversion as I, metrics as MT, observations as O

sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module
AB = import_module("01_ablation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ne", type=int, default=200)
    ap.add_argument("--na", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rtps", type=float, default=0.7)
    args = ap.parse_args()

    tr, mask_t, mask_e, geom = AB.load_setup()
    q_true = tr["q_annual"]
    pr = E.prior(mask_e, C.EST)
    X = E.sample_prior(pr, args.ne)
    root = ROOT / "runs" / "ens"

    legs = tuple(k for k in I.LEGS if k != "meter")   # satellites only, no meters
    obs = np.concatenate([tr["obs_" + k] for k in legs])
    sd = np.concatenate([tr["sig_" + k] for k in legs])
    full = AB.leg_index(tr)
    take = np.concatenate([np.arange(full[k].start, full[k].stop) for k in legs])
    idx, off = {}, 0
    for k in legs:
        n = tr["obs_" + k].size
        idx[k] = slice(off, off + n)
        off += n
    rho = I.taper(geom, C.EST, legs=legs)
    rng = np.random.default_rng(97)
    alphas = I.alpha_schedule(args.na)

    hist = []
    Dfull, ok = I.run_ensemble(X, root, C.EST, mask_e, geom, workers=args.workers)
    D = Dfull[take]
    for it in range(args.na + 1):
        m = D[:, ok].mean(axis=1)
        row = {"iter": it}
        for k in legs:
            r = m[idx[k]] - obs[idx[k]]
            row[k] = float(np.sqrt((r ** 2).mean()))
            row[k + "_chi"] = float((((D[idx[k]][:, ok] - obs[idx[k], None])
                                     / sd[idx[k], None]) ** 2).mean())
        ens = np.array([E.q_annual(X[:, i]) for i in np.nonzero(ok)[0]])
        s = MT.point_scores(ens.mean(axis=0), q_true)
        s.update(MT.coverage(ens, q_true))
        row.update({k: float(v) for k, v in s.items()})
        hist.append(row)
        print(f"it{it}  MAE {row['mae_mcm']:7.2f}  bias {row['bias_mcm']:7.2f}  "
              f"cov90 {row['cover_90']:.2f} | resid et {row['et']:.3g} "
              f"grace {row['grace']:.3g} insar {row['insar']:.4g} head {row['head']:.3g}")
        if it == args.na:
            break
        X = I.esmda_update(X, D, obs, sd, float(alphas[it]), rho, rng, ok, pr=pr, rtps=args.rtps)
        Dfull, ok = I.run_ensemble(X, root, C.EST, mask_e, geom, workers=args.workers)
        D = Dfull[take]

    (ROOT / "results" / "convergence.json").write_text(json.dumps(hist, indent=2))
    np.savez_compressed(ROOT / "results" / "convergence.npz", X=X, ok=ok)
    print("wrote results/convergence.json")


if __name__ == "__main__":
    main()
