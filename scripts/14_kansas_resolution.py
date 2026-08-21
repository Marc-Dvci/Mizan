"""What the Kansas observations resolved, against the null of no data.

The same generalised-eigenvalue analysis L0 publishes, run on the L2 posteriors. It is
computed from the prior and posterior ensembles alone, so it is available before the
meters are opened, and it is what says in advance whether county-year abstraction is
identifiable in that basin from those legs.

`scripts/13_null.py` has to have run first; its L2 row is what these numbers are read
against.

    python scripts/14_kansas_resolution.py [--tag ""]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mizan import ks_data as K, ks_run as R, metrics as MT  # noqa: E402

RES = ROOT / "results"
LABEL = {"ET": "evapotranspiration only", "H": "heads only",
         "ETH": "evapotranspiration + heads, closure"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--ne", type=int, default=250)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", type=str, default="kansas_resolution.json")
    args = ap.parse_args()

    region = K.build_region(K.load_points())
    irr = K.irrigated_area(K.irrigated_fraction(region), region)
    pr = R.prior(region, irr)
    X0 = R.sample_prior(pr, args.ne, seed=args.seed)
    prior_ens = (10.0 ** X0[R.LAYOUT["logq"]]).T.reshape(args.ne, R.NDIST, R.NYEAR)

    out = {"_ndir": R.NDIST * R.NYEAR, "_ne": args.ne}
    nullp = RES / "resolution_null.json"
    if nullp.exists():
        n = json.loads(nullp.read_text())["L2"]
        out["_null"] = {k: n[k]["mean"] for k in
                        ("n_resolved_90", "n_unresolved", "n_widened", "effective_dim")}

    for k in ("ET", "H", "ETH"):
        p = RES / f"kansas_posterior_{k}{args.tag}.npz"
        if not p.exists():
            continue
        d = MT.resolved_directions(np.load(p)["ens"], prior_ens)
        out[k] = {"label": LABEL[k],
                  "n_resolved_90": d["n_resolved_90"],
                  "n_resolved_50": d["n_resolved_50"],
                  "n_unresolved": d["n_unresolved"],
                  "n_widened": d["n_widened"],
                  "effective_dim": d["effective_dim"]}

    (RES / args.out).write_text(json.dumps(out, indent=2))

    nl = out.get("_null", {})
    print(f"{'configuration':40s} {'resolved':>9s} {'unresolved':>11s} {'widened':>8s} "
          f"{'constrained':>12s} {'above null':>11s}")
    if nl:
        print(f"{'null: no data at all':40s} {nl['n_resolved_90']:9.0f} "
              f"{nl['n_unresolved']:11.0f} {nl['n_widened']:8.0f} "
              f"{nl['effective_dim']:12.1f} {0.0:11.1f}")
    for k in ("ET", "H", "ETH"):
        if k not in out:
            continue
        v = out[k]
        gain = v["effective_dim"] - nl.get("effective_dim", 0.0)
        print(f"{v['label']:40s} {v['n_resolved_90']:9d} {v['n_unresolved']:11d} "
              f"{v['n_widened']:8d} {v['effective_dim']:12.1f} {gain:+11.1f}")
    print(f"\nout of {out['_ndir']} county-year directions")


if __name__ == "__main__":
    main()
