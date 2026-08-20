"""Generate the L0 truth: run the fine-grid reality, then synthesise observations.

Outputs `results/truth.npz`, which is the only file the inversion is allowed to read.
"""
from __future__ import annotations

import argparse
import sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mizan import config as C, forcing as F, model as M, observations as O, truth as T

WS = ROOT / "runs" / "truth"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=str, default="",
                    help="district:Mm3/yr of abstraction with no canopy and therefore "
                         "no evapotranspiration signature, e.g. 3:40")
    ap.add_argument("--out", type=str, default="truth.npz")
    ap.add_argument("--eta-uniform", action="store_true",
                    help="give every district the same consumptive fraction, equal to "
                         "the mean of the district values. This is the case most "
                         "favourable to the open-loop account, whose whole error in the "
                         "reference truth comes from assuming one constant for all.")
    ap.add_argument("--ws", type=str, default="",
                    help="scratch directory name under runs/, default derived from --out")
    args = ap.parse_args()

    if args.eta_uniform:
        C.DIST_ETA = np.full(C.NDIST, float(C.DIST_ETA.mean()))
        print(f"uniform consumptive fraction: eta = {C.DIST_ETA[0]:.4f} everywhere")
    hidden_d, hidden_v = (-1, 0.0)
    if args.hidden:
        a, b = args.hidden.split(":")
        hidden_d, hidden_v = int(a), float(b) * 1.0e6

    grid = C.TRUTH
    mask = F.pivot_mask(grid)
    p = T.params(grid)
    if hidden_d >= 0:
        days = np.tile(np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
                                dtype=float), C.NYEAR)
        p.q_monthly = p.q_monthly.copy()
        p.q_monthly[hidden_d] += hidden_v / 365.25
        print(f"hidden withdrawal: district {hidden_d}, {hidden_v/1e6:.0f} Mm3/yr, "
              f"no canopy and therefore no evapotranspiration signature")
    print(f"truth grid {grid.ncol}x{grid.ncol} @ {grid.delr_m:.0f} m, "
          f"{int(mask.sum())} irrigated cells")

    t0 = time.time()
    ws = Path(str(WS) + "_" + args.ws) if args.ws else (
        WS if hidden_d < 0 else Path(str(WS) + "_hidden"))
    M.build(ws, grid, p, mask, inelastic=True)
    ok = M.run(ws)
    print(f"forward run {time.time()-t0:.1f}s, normal termination: {ok}")
    if not ok:
        raise SystemExit("truth run failed")

    st = O.read_state(ws, grid)
    geom = O.make_geometry(mask, grid)

    ann_visible = F.district_annual_abstraction()
    ann = ann_visible.copy()
    if hidden_d >= 0:
        ann[hidden_d] += hidden_v
    clean = {
        "et": O.op_et(ann_visible, C.DIST_ETA, C.DIST_PREPLANT),
        "grace": O.op_grace(st, T.GRACE_ALPHA, T.GRACE_DRIFT),
        "insar": O.op_insar(st, geom, grid, T.INSAR_RAMP),
        "head": O.op_head(st, geom, grid),
        "meter": O.op_meter(ann),
    }
    rng = np.random.default_rng(2026)
    noisy = O.add_noise(clean, rng)
    sig = O.obs_sigma(clean)

    inel = O.read_inelastic(ws, grid)
    perm = float(inel[-1].sum() * grid.delr_m ** 2)

    subs = st.subsidence
    print(f"peak subsidence {subs[-1].max()*100:.1f} cm over {C.NYEAR} yr "
          f"({subs[-1].max()*100/C.NYEAR:.2f} cm/yr)")
    print(f"max head decline {(C.H_INIT - st.head[-1, 1]).max():.1f} m")
    print(f"storage depletion {st.storage_depletion[-1]/1e9:.2f} km3 over {C.NYEAR} yr")
    print(f"total abstraction {ann.sum()/1e9:.2f} km3, mean {ann.sum(0).mean()/1e6:.0f} Mm3/yr")
    print(f"storage capacity destroyed permanently {perm/1e9:.3f} km3, "
          f"{perm/st.storage_depletion[-1]*100:.1f}% of what was removed")
    print(f"obs counts: " + ", ".join(f"{k}={v.size}" for k, v in noisy.items()))

    np.savez_compressed(
        ROOT / "results" / args.out,
        q_annual=ann, q_visible=ann_visible, hidden_district=hidden_d,
        eta=C.DIST_ETA, preplant=C.DIST_PREPLANT,
        well_xy=geom.well_xy, well_seen=geom.well_seen,
        insar_xy=geom.insar_xy, insar_ref_xy=geom.insar_ref_xy,
        subsidence_final=subs[-1], head_final=st.head[-1, 1],
        storage_depletion=st.storage_depletion, permanent_loss=perm,
        inelastic_final=inel[-1].sum(axis=0),
        **{f"obs_{k}": v for k, v in noisy.items()},
        **{f"clean_{k}": v for k, v in clean.items()},
        **{f"sig_{k}": v for k, v in sig.items()},
    )
    print(f"wrote results/{args.out}")


if __name__ == "__main__":
    main()
