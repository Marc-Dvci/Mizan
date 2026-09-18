"""The two comparisons the entry rests on, each with the error bar its claim requires.

A ranking of two point estimates is not a result until the record says how well it
resolves the difference between them, and the answer depends on what is being claimed.

**Two error bars, two questions.** Clustering by year asks how the gain would come out in
another year over these same six counties. Clustering by county asks whether it would
hold in a district this record has not seen, which is the question a transfer claim
actually asks, and its effective sample is the number of counties, six, not the number
of county-years. The county figure is much the wider of the two and it is the one
reported first, because it is the claim being made.

Two comparisons, and they do not resolve equally well.

* **The level, against the published method.** Open-loop satellite accounting,
  consumptive use divided by an assumed efficiency, is the technique in use where wells
  are not metered. The closure removes about twenty points of relative error from it
  over the block, but the gain is carried by the counties where that account fails worst
  and is slightly negative in two of the six, so clustered by county it is under two
  standard errors.
* **The change between two multi-year periods, against every meter-free account.** This
  is the quantity a reduction target is written in, and it is the comparison the record
  resolves worst: the window pairs overlap, and the metered era contains exactly one
  pair of five-year windows sharing no year with another such pair.

    python scripts/29_headline.py [--tag _v3]

Writes results/headline{tag}.json.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mizan import ks_data as K, ks_run as R

RES = ROOT / "results"
LABEL = {
    "OPENLOOP": "the published open-loop account, consumptive use over a fixed 0.80",
    "OPENLOOP_ORACLE": "the same, with its efficiency fitted to the withheld meters",
    "FLAT": "mapped irrigated area times one acre-foot per acre",
    "WATERBAL": "the same, plus half the year's precipitation deficit",
    "CLOSURE": "the closure, evapotranspiration and heads",
}


def jackknife(fn, nyear: int):
    """Observed value of `fn` over all years, its jackknife error, and its extremes."""
    obs = fn(np.arange(nyear))
    vals = np.array([fn(np.array([t for t in range(nyear) if t != d]))
                     for d in range(nyear)])
    n = vals.size
    se = float(np.sqrt((n - 1) / n * ((vals - vals.mean()) ** 2).sum()))
    return float(obs), se, float(vals.min()), float(vals.max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="_v3")
    ap.add_argument("--era", type=int, default=K.METERED_ERA_YEAR0)
    args = ap.parse_args()

    sweep = importlib.import_module("27_metered_era")
    drv = importlib.import_module("11_kansas_run")
    a = drv.assemble(pool=True)
    years = np.arange(K.YEAR0, K.YEAR1 + 1)
    era = years >= args.era
    q_all = K.reported_annual()[0].sum(axis=0)
    q = q_all[:, era]
    ens = np.load(RES / f"kansas_posterior_ETH{args.tag}.npz")["ens"][..., era]
    hat = ens.mean(axis=0)

    P = K.precipitation()
    deficit = P.mean(axis=1, keepdims=True) - P
    et = a["et_obs"][:, era]
    grid = np.linspace(0.4, 1.2, 801)
    e_star = float(grid[int(np.argmin([np.abs(et / g - q).mean() for g in grid]))])
    acc = {
        "CLOSURE": hat,
        "OPENLOOP": et / 0.80,
        "OPENLOOP_ORACLE": et / e_star,
        "FLAT": (a["irr_area"] * R.PRIOR_DEPTH_M)[:, era],
        "WATERBAL": (a["irr_area"] * (R.PRIOR_DEPTH_M + 0.5 * deficit / 1000.0))[:, era],
    }

    def mape(v, cols):
        return float((np.abs(v[:, cols] - q[:, cols]) / q[:, cols]).mean() * 100.0)

    out = {"_meta": {"tag": args.tag, "era_year0": int(args.era),
                     "n_county_years": int(q.size), "n_years": int(q.shape[1]),
                     "n_counties": len(K.COUNTIES),
                     "oracle_efficiency": e_star,
                     "error_bars": ("by county: the unit a transfer claim generalises "
                                    "over, n=6; by year: leave-one-year-out over the "
                                    "metered era, n=16")}}

    out["level"] = {}
    for k, v in acc.items():
        out["level"][k] = {
            "label": LABEL[k],
            "mape_pct": mape(v, np.arange(q.shape[1])),
            "mae_mcm": float(np.abs(v - q).mean() / 1e6),
            "basin_bias_pct": float((v.sum() - q.sum()) / q.sum() * 100.0),
        }

    out["level_gain_vs"] = {}
    for k in ("OPENLOOP", "OPENLOOP_ORACLE", "FLAT", "WATERBAL"):
        obs, se_y, lo, hi = jackknife(
            lambda c, k=k: mape(acc[k], c) - mape(acc["CLOSURE"], c), q.shape[1])
        per_county = ((np.abs(acc[k] - q) / q).mean(axis=1)
                      - (np.abs(acc["CLOSURE"] - q) / q).mean(axis=1)) * 100.0
        nc = per_county.size
        se_c = float(per_county.std(ddof=1) / np.sqrt(nc))
        out["level_gain_vs"][k] = {
            "label": LABEL[k],
            "points": obs,
            "se_by_county": se_c,
            "n_se_by_county": obs / se_c if se_c > 0 else float("nan"),
            "se_by_year": se_y,
            "n_se_by_year": obs / se_y if se_y > 0 else float("nan"),
            "worst_year_drop": lo,
            "best_year_drop": hi,
            "gain_by_county": {c: round(float(x), 1)
                               for c, x in zip(K.COUNTIES, per_county)},
            "n_counties_favouring_closure": int((per_county > 0).sum()),
            "n_counties": int(nc),
            "closure_closer_pct_of_county_years": float(
                100.0 * (np.abs(acc["CLOSURE"] - q) < np.abs(acc[k] - q)).mean()),
        }

    # The change comparison, at every window, on the era and on the whole record.
    pts_all = {"FLAT": a["irr_area"] * R.PRIOR_DEPTH_M,
               "WATERBAL": a["irr_area"] * (R.PRIOR_DEPTH_M + 0.5 * deficit / 1000.0),
               "OPENLOOP": a["et_obs"] / 0.80}
    ens_all = np.load(RES / f"kansas_posterior_ETH{args.tag}.npz")["ens"]

    def margin_at(E, PT, Q, Y, w):
        def f(cols):
            s, _, _, _ = sweep.sweep(E[..., cols],
                                     {k: v[:, cols] for k, v in PT.items()},
                                     Q[:, cols], Y[cols], w)
            m = s["mean_abs_error_pts"]
            return min(m[k] for k in PT) - m["CLOSURE"]
        return f

    out["change_margin"] = {}
    for name, (E, PT, Q, Y) in {
            "metered_era": (ens, {k: v[:, era] for k, v in pts_all.items()},
                            q, years[era]),
            "whole_record": (ens_all, pts_all, q_all, years)}.items():
        rows = {}
        for w in range(4, 9):
            if len(Y) < 2 * w + 1:
                continue
            s, _, _, _ = sweep.sweep(E, PT, Q, Y, w)
            try:
                obs, se, lo, hi = jackknife(margin_at(E, PT, Q, Y, w), len(Y))
            except ValueError:
                continue
            rows[str(w)] = {"n_pairs": s["n_pairs"], "points": obs, "se_by_year": se,
                            "n_se_by_year": obs / se if se > 0 else float("nan"),
                            "worst_year_drop": lo,
                            "sign_survives_any_single_year_drop": bool(lo > 0)}
        out["change_margin"][name] = rows

    pp = sweep.pairs_of(int(era.sum()), 5)
    used = [set(range(i, i + 5)) | set(range(j, j + 5)) for i, j in pp]
    chosen: list[int] = []
    for idx, u in enumerate(used):
        if all(not (u & used[c]) for c in chosen):
            chosen.append(idx)
    out["change_margin"]["independent_five_year_pairs_in_era"] = len(chosen)

    (RES / f"headline{args.tag}.json").write_text(json.dumps(out, indent=2))

    nl = chr(10)
    print("Level, {} county-years over {} to {}{}".format(
        q.size, args.era, K.YEAR1, nl))
    print("  {:<58s} {:>8s} {:>10s}".format("account", "rel err", "MAE Mm3/yr"))
    for k in ("OPENLOOP", "OPENLOOP_ORACLE", "WATERBAL", "FLAT", "CLOSURE"):
        v = out["level"][k]
        print("  {:<58s} {:7.1f}% {:10.2f}".format(v["label"][:58], v["mape_pct"],
                                                   v["mae_mcm"]))

    print(nl + "Points of relative error the closure removes, and how well each is "
          "resolved:")
    print("  {:<20s} {:>7s} {:>18s} {:>18s} {:>12s}".format(
        "", "gain", "by county (n=6)", "by year (n=16)", "counties +"))
    for k, v in out["level_gain_vs"].items():
        print("  {:<20s} {:+7.1f} {:>10.1f} ({:+.1f} se) {:>10.1f} ({:+.1f} se) "
              "{:>8d}/{:d}".format(
                  k, v["points"], v["se_by_county"], v["n_se_by_county"],
                  v["se_by_year"], v["n_se_by_year"],
                  v["n_counties_favouring_closure"], v["n_counties"]))
    print("  The county column is what a transfer claim needs: its sample is the number")
    print("  of districts, not of district-years, and six counties of one block share a")
    print("  climate, a retrieval and a reporting cycle.")
    for k, v in out["level_gain_vs"].items():
        print("  {:<20s} by county {}".format(k, v["gain_by_county"]))

    print(nl + "Margin over the best meter-free bar on the change, by window:")
    for name, rows in out["change_margin"].items():
        if not isinstance(rows, dict):
            continue
        print("  " + name)
        for w, v in rows.items():
            print("    {:>2s} yr  {:4d} pairs  {:+6.2f} +- {:5.2f} by year ({:+.1f} se)"
                  "  sign survives any one-year drop: {}".format(
                      w, v["n_pairs"], v["points"], v["se_by_year"], v["n_se_by_year"],
                      v["sign_survives_any_single_year_drop"]))
    print("  five-year window pairs in the era sharing no year with another: {}".format(
        out["change_margin"]["independent_five_year_pairs_in_era"]))

    print(nl + "wrote results/headline{}.json".format(args.tag))


if __name__ == "__main__":
    main()
