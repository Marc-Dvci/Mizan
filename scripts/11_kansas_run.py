"""L2 Kansas: run the closure on real observations and score it against the meters.

    python scripts/11_kansas_run.py --ne 200 --na 6

Writes `results/kansas.json` and `results/kansas_posterior.npz`. The meters are read
once, at the end, to score. Nothing upstream of the scoring touches them.
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

from mizan import inversion as I, ks_data as K, ks_run as R, metrics as MT

RES = ROOT / "results"


def assemble() -> dict:
    pts = K.load_points()
    region = K.build_region(pts)
    et = K.evapotranspiration(region)
    frac = K.irrigated_fraction(region)
    et_obs, et_se = K.irrigation_et(et, frac, region)
    irr_area = K.irrigated_area(frac, region)
    wl = K.water_levels(region)
    ctx = R.make_context(region, wl)
    return dict(region=region, et=et, frac=frac, et_obs=et_obs, et_se=et_se,
                irr_area=irr_area, wl=wl, ctx=ctx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ne", type=int, default=250)
    ap.add_argument("--na", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--rtps", type=float, default=0.7)
    ap.add_argument("--rows", type=str, default="ETH,ET,H",
                    help="ETH both legs, ET evapotranspiration only, H heads only")
    ap.add_argument("--out", type=str, default="kansas.json")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    a = assemble()
    region, ctx = a["region"], a["ctx"]
    n_et = R.NDIST * R.NYEAR
    obs = np.concatenate([a["et_obs"].ravel(), R.head_anomaly(a["wl"], ctx)])
    # The unmixing returns the standard error of its own slope, so a county with almost
    # no irrigated ground enters the likelihood at the weight its leverage supports.
    et_sd = np.maximum(np.hypot(a["et_se"], R.ET_REL_SIGMA * a["et_obs"]),
                       0.03 * a["et_obs"].max()).ravel()
    sd = np.concatenate([et_sd, np.full(obs.size - n_et, R.HEAD_SIGMA_M)])

    irr_km2 = a["irr_area"].mean(axis=1).sum() / 1e6
    print(f"region {region.nrow}x{region.ncol} at {region.delr_m/1000:.0f} km, "
          f"{int((region.county >= 0).sum())} active cells")
    print(f"irrigated {irr_km2:,.0f} km2 of "
          f"{(region.county >= 0).sum() * region.area_m2 / 1e6:,.0f} km2")
    print(f"observations: {n_et} county-year evapotranspiration, "
          f"{obs.size - n_et} well-year head anomalies from {len(a['wl']['wells'])} wells")

    pr = R.prior(region, a["irr_area"])
    X0 = R.sample_prior(pr, args.ne, seed=args.seed)
    root = ROOT / "runs" / ("ks" + args.tag)
    rho_full = R.taper(ctx)

    t0 = time.time()
    D0, ok0 = R.run_ensemble(X0, root, ctx, workers=args.workers)
    print(f"prior ensemble {time.time()-t0:.0f}s, {int(ok0.sum())}/{args.ne} converged")

    q_true, meta = K.metered_annual()
    print(f"withheld truth: {meta['n_rights']} water rights, "
          f"{meta['n_missing']} unreadable, "
          f"{q_true.sum(axis=0).mean()/1e6:,.0f} Mm3/yr over the block")

    take = {"ETH": np.arange(obs.size),
            "ET": np.arange(n_et),
            "H": np.arange(n_et, obs.size)}

    res = json.loads((RES / args.out).read_text()) if (RES / args.out).exists() else {}
    res["_meta"] = {"seed": args.seed, "ne": args.ne, "na": args.na,
                    "irrigated_km2": irr_km2, "n_wells": len(a["wl"]["wells"]),
                    "n_obs_head": int(obs.size - n_et), "counties": K.COUNTIES,
                    "years": [K.YEAR0, K.YEAR1], **meta}

    # The published open-loop account, on the same evapotranspiration the closure sees.
    ol = a["et_obs"] / 0.80
    res["BASELINE"] = {"label": "open loop, efficiency fixed at 0.80",
                       **MT.point_scores(ol, q_true)}
    # The same oracle definition as L0: the single constant that minimises absolute
    # error against the answer. No practitioner has it; it exists so the comparison is
    # against the best the open-loop form can do rather than against its constant.
    grid = np.linspace(0.20, 1.60, 1401)
    err = np.array([np.abs(a["et_obs"] / g - q_true).mean() for g in grid])
    e_star = float(grid[int(err.argmin())])
    res["BASELINE_ORACLE"] = {
        "label": f"open loop, efficiency fitted to the meters at {e_star:.3f}",
        **MT.point_scores(a["et_obs"] / e_star, q_true)}
    ps = MT.point_scores(np.array([R.decode(X0[:, i])["q"]
                                   for i in np.nonzero(ok0)[0]]).mean(axis=0), q_true)
    res["PRIOR"] = {"label": "prior, no data at all", **ps}

    for key in args.rows.split(","):
        idx = take[key]
        rng = np.random.default_rng(97 + 1000 * (args.seed - 5))
        X, D, ok = X0.copy(), D0[idx], ok0
        rho = rho_full[:, idx]
        alphas = I.alpha_schedule(args.na)
        hist = []
        t = time.time()
        for it in range(args.na):
            r = (D - obs[idx][:, None]) / sd[idx][:, None]
            hist.append(float((r[:, ok] ** 2).mean()))
            X = I.esmda_update(X, D, obs[idx], sd[idx], float(alphas[it]), rho, rng,
                               ok, pr=pr, rtps=args.rtps)
            Dfull, ok = R.run_ensemble(X, root, ctx, workers=args.workers)
            D = Dfull[idx]
        r = (D - obs[idx][:, None]) / sd[idx][:, None]
        hist.append(float((r[:, ok] ** 2).mean()))

        ens = np.array([R.decode(X[:, i])["q"] for i in np.nonzero(ok)[0]])
        sc = MT.point_scores(ens.mean(axis=0), q_true)
        sc.update(MT.coverage(ens, q_true))
        sc["crps_mcm"] = MT.crps(ens, q_true)
        res[key] = {"label": {"ETH": "evapotranspiration + heads, closure",
                              "ET": "evapotranspiration only",
                              "H": "heads only"}[key],
                    **sc, "phi_history": hist,
                    "eta_hat": X[R.LAYOUT["eta"]][:, ok].mean(axis=1).tolist(),
                    "sy_hat": float((10.0 ** X[R.LAYOUT["log_sy"]][:, ok]).mean()),
                    "bsat_hat": float((10.0 ** X[R.LAYOUT["log_bsat"]][:, ok]).mean()),
                    "rch_hat": float((10.0 ** X[R.LAYOUT["log_rch"]][:, ok]).mean()),
                    "seconds": time.time() - t}
        np.savez_compressed(RES / f"kansas_posterior_{key}{args.tag}.npz",
                            X=X, ok=ok, ens=ens, q_true=q_true, et_obs=a["et_obs"],
                            county=region.county, frac=a["frac"],
                            et_se=a["et_se"], irr_area=a["irr_area"])
        print(f"  {key:4s} MAE {sc['mae_mcm']:7.2f} Mm3/yr  MAPE {sc['mape_pct']:5.1f}%  "
              f"cover90 {sc.get('cover_90', float('nan')):.2f}  ({time.time()-t:.0f}s)")

    (RES / args.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote results/{args.out}")
    for k, v in res.items():
        if k.startswith("_"):
            continue
        print(f"{k:5s} {v['label']:44s} {v['mae_mcm']:8.2f} {v['mape_pct']:6.1f}%")


if __name__ == "__main__":
    main()
