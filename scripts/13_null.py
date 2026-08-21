"""The null of the resolution statistic, at L0 and at L2.

`metrics.resolved_directions` compares a posterior ensemble against the prior ensemble
it came from and reports how many directions of the abstraction vector the data
resolved. With a few hundred members and a hundred and fifty or more directions, the
sample covariances are noisy enough that the statistic is not zero when nothing has been
learned, so the number it returns cannot be read without knowing what it returns for no
information at all.

This measures that. Two independent draws from the same prior, no data assimilated, no
forward model run. Everything a row reports has to be read against this row.

    python scripts/13_null.py [--ne 250] [--pairs 6]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mizan import (config as C, estimator as E, fields, forcing as F,  # noqa: E402
                   ks_data as K, ks_run as R, metrics as MT)

RES = ROOT / "results"
KEYS = ("n_resolved_90", "n_resolved_50", "n_unresolved", "n_widened", "effective_dim")


def l0_draw(pr, ne: int, seed: int) -> np.ndarray:
    X = E.sample_prior(pr, ne, seed=seed)
    return np.array([E.q_annual(X[:, i]) for i in range(ne)])


def ks_draw(pr, ne: int, seed: int) -> np.ndarray:
    X = R.sample_prior(pr, ne, seed=seed)
    return (10.0 ** X[R.LAYOUT["logq"]]).T.reshape(ne, R.NDIST, R.NYEAR)


def null_for(draw, pr, ne: int, pairs: int) -> dict:
    """Resolution statistic between independent prior draws, over several seed pairs."""
    base = draw(pr, ne, 5)
    rows = [MT.resolved_directions(draw(pr, ne, s), base)
            for s in range(6, 6 + pairs)]
    out = {}
    for k in KEYS:
        v = np.array([float(r[k]) for r in rows])
        out[k] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
    out["ne"] = ne
    out["pairs"] = pairs
    out["ndir"] = int(base.shape[1] * base.shape[2])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ne", type=int, default=250)
    ap.add_argument("--pairs", type=int, default=6)
    args = ap.parse_args()

    mask_e = fields.upscale_mask(F.pivot_mask(C.TRUTH),
                                 int(C.EST.delr_m // C.TRUTH.delr_m))
    l0 = null_for(l0_draw, E.prior(mask_e, C.EST), args.ne, args.pairs)

    region = K.build_region(K.load_points())
    irr = K.irrigated_area(K.irrigated_fraction(region), region)
    ks = null_for(ks_draw, R.prior(region, irr), args.ne, args.pairs)

    out = {"L0": l0, "L2": ks}
    (RES / "resolution_null.json").write_text(json.dumps(out, indent=2))

    for name, d in out.items():
        print(f"{name}: {d['ndir']} directions, ne={d['ne']}, {d['pairs']} independent "
              f"prior pairs, no data assimilated")
        for k in KEYS:
            v = d[k]
            print(f"    {k:16s} {v['mean']:8.1f}   [{v['min']:.1f}, {v['max']:.1f}]")


if __name__ == "__main__":
    main()
