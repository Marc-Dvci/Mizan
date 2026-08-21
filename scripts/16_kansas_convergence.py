"""Is a failed ensemble member a random member, or a heavily pumped one?

Putting the published saturated thickness under the Kansas model takes the layer from
79 m to about 18 m, and a thin unconfined layer is harder to solve: the prior ensemble
converges for fewer members than it did. That is only a cost in sample size if the
members that fail are drawn like the ones that survive. If instead the solver drops the
heavily pumped draws, the prior has been truncated from above by a numerical accident
and every posterior built on it is biased low, with nothing in the score to show it.

This runs the prior forward once and compares the two groups on the quantity that
matters, which is the abstraction the member was drawn with.

    python scripts/16_kansas_convergence.py [--ne 60] [--workers 3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mizan import ks_data as K, ks_run as R

RES = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ne", type=int, default=60)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    pts = K.load_points()
    region = K.build_region(pts)
    frac = K.irrigated_fraction(region)
    irr_area = K.irrigated_area(frac, region)
    wl = K.water_levels(region)
    ctx = R.make_context(region, wl)

    pr = R.prior(region, irr_area)
    X = R.sample_prior(pr, args.ne, seed=args.seed)
    _, ok = R.run_ensemble(X, ROOT / "runs" / "ks_conv", ctx, workers=args.workers)

    sl = R.LAYOUT["logq"]
    q = (10.0 ** X[sl]).reshape(len(K.COUNTIES), -1, args.ne).sum(axis=0).mean(axis=0) / 1e6
    bm = 10.0 ** X[R.LAYOUT["log_bmul"]][0]
    sy = 10.0 ** X[R.LAYOUT["log_sy"]][0]

    n_ok, n_bad = int(ok.sum()), int((~ok).sum())
    print(f"{n_ok} of {args.ne} converged\n")
    if n_bad == 0:
        print("nothing failed, so there is nothing to be biased")
        return

    rows = {"block abstraction, Mm3/yr": q,
            "saturated thickness multiplier": bm,
            "specific yield": sy}
    out = {"ne": args.ne, "n_ok": n_ok, "n_failed": n_bad}
    print(f"{'quantity':34s} {'converged':>12s} {'failed':>12s} {'gap in sd':>11s}")
    for name, v in rows.items():
        a, b = v[ok], v[~ok]
        sd = v.std()
        z = (b.mean() - a.mean()) / sd if sd > 0 else 0.0
        out[name] = {"converged": float(a.mean()), "failed": float(b.mean()),
                     "gap_sd": float(z)}
        print(f"{name:34s} {a.mean():12.3f} {b.mean():12.3f} {z:+11.2f}")

    z = abs(out["block abstraction, Mm3/yr"]["gap_sd"])
    print(f"\nThe failed members sit {z:.2f} standard deviations from the converged ones "
          f"on abstraction.")
    print("Below about half a standard deviation the loss is sample size. Above it the "
          "prior has been truncated by the solver and the posterior inherits it.")
    (RES / "kansas_convergence.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
