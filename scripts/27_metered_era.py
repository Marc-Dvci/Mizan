"""L2 Kansas: score every account again on the years the truth is fully metered.

WIMAS carries a measurement code on every water-use report (KGS OFR 2005-30). Three
codes mean a meter was read, one means the volume was computed from hours of pump
operation and a rate. In the six counties the meter-coded share of reported volume is
24 to 29 per cent for 2000 to 2005, rises through 2006 to 2008, and is 98.8 per cent or
more in every year from 2009, the year GMD4 records as the first with every well
metered. The published `_v3` scores were computed against the whole 2000 to 2024
record labelled as metered. This script computes them on the metered era only.

Two things change at once and they are kept apart:

* the **truth series** is rebuilt from the WIMAS use file, which files one report per
  point of diversion. The published series read one point per water right from the
  history page and so under-counts rights with several reporting points, by 7 to 10
  per cent of the block volume in every year;
* the **years** are restricted to 2009 to 2024 for every score that is called metered.

Nothing on the estimate side is re-run: the posteriors are the published ones and
never saw a meter. Only the scoring changes.

    python scripts/27_metered_era.py --tag _v3

Writes results/metered_era{tag}.json and figures/fig16_metered_era.png.
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

from mizan import ks_data as K, ks_run as R, metrics as MT

RES = ROOT / "results"
FIG = ROOT / "figures"

W = 5
ACCOUNTS = ("CLOSURE", "FLAT", "WATERBAL", "OPENLOOP")
LABEL = {"CLOSURE": "closure, evapotranspiration + heads",
         "FLAT": "mapped irrigated area x one acre-foot per acre",
         "WATERBAL": "the same, plus half the year's precipitation deficit",
         "OPENLOOP": "unmixed evapotranspiration over a fixed efficiency of 0.80"}

# Fixed by the policy record, as in 19_verify.py. Only the GMD4 LEMA contrast lies
# inside the metered era. The era's own long contrast is its first five years against
# its last five, which is set by the era's endpoints and not by any score.
CONTRASTS = [
    ("GMD4", "the district-wide GMD4 LEMA period against the five years before it",
     (2013, 2017), (2018, 2022)),
    ("ERA", "the last five years of the metered era against its first five",
     (2009, 2013), (2020, 2024)),
]


def contrast(v, years, a, b):
    A = (years >= a[0]) & (years <= a[1])
    B = (years >= b[0]) & (years <= b[1])
    return 100.0 * (v[..., B].mean(-1).sum(-1) / v[..., A].mean(-1).sum(-1) - 1.0)


def pairs_of(n: int, w: int):
    st = range(0, n - w + 1)
    return [(i, j) for i in st for j in st if j >= i + w]


def sweep(ens, points, q_true, years, w):
    """Mean absolute error on the change over every non-overlapping w-year pair."""
    pp = pairs_of(len(years), w)
    err = {k: [] for k in ACCOUNTS}
    cover = declared = right = 0
    t_all, e_all, flag = [], [], []
    for i, j in pp:
        aw = (years[i], years[i + w - 1])
        bw = (years[j], years[j + w - 1])
        t = float(contrast(q_true, years, aw, bw))
        e = contrast(ens, years, aw, bw)
        err["CLOSURE"].append(abs(float(e.mean()) - t))
        lo, hi = np.percentile(e, 5), np.percentile(e, 95)
        cover += int(lo <= t <= hi)
        f = not (lo <= 0.0 <= hi)
        if f:
            declared += 1
            right += int(np.sign(e.mean()) == np.sign(t))
        t_all.append(t)
        e_all.append(float(e.mean()))
        flag.append(f)
        for k, v in points.items():
            err[k].append(abs(float(contrast(v, years, aw, bw)) - t))
    n = len(pp)
    t_all, e_all, flag = np.array(t_all), np.array(e_all), np.array(flag)
    out = {"n_pairs": n,
           "mean_abs_error_pts": {k: float(np.mean(v)) for k, v in err.items()},
           "median_abs_error_pts": {k: float(np.median(v)) for k, v in err.items()},
           "closure_beats_pct": {k: float(100.0 * np.mean(np.array(err["CLOSURE"])
                                                          < np.array(err[k])))
                                 for k in points},
           "coverage_90": float(cover / n),
           "n_declared_change": int(declared),
           "n_declared_change_sign_correct": int(right),
           "n_metered_change_negative": int((t_all < 0).sum())}
    if flag.any() and (~flag).any():
        big, small = np.abs(t_all)[flag], np.abs(t_all)[~flag]
        out["declaration_auc_vs_metered_magnitude"] = float(
            (big[:, None] > small[None, :]).mean()
            + 0.5 * (big[:, None] == small[None, :]).mean())
        out["metered_pct_where_declared"] = float(big.mean())
        out["metered_pct_where_not_declared"] = float(small.mean())
    if n > 2:
        out["change_r"] = float(np.corrcoef(e_all, t_all)[0, 1])
    return out, err, t_all, e_all


def level(ens, points, q_true):
    """Level scores of every account, district-year, against one truth."""
    out = {"CLOSURE": {**MT.point_scores(ens.mean(axis=0), q_true),
                       **MT.coverage(ens, q_true),
                       "crps_mcm": MT.crps(ens, q_true)}}
    for k, v in points.items():
        out[k] = MT.point_scores(v, q_true)
    return out


def anomaly(x):
    return x - x.mean(axis=1, keepdims=True)


def loco_skill(hat, q_true):
    """Interannual skill with a leave-one-county-out amplitude, as 15_kansas_shrink."""
    grid = np.linspace(0.0, 1.5, 151)
    ah, at = anomaly(hat), anomaly(q_true)
    signal = float(np.abs(at).mean())

    def best(rows):
        err = np.array([np.abs(s * ah[rows] - at[rows]).mean() for s in grid])
        return float(grid[int(err.argmin())])

    nc = ah.shape[0]
    fac = np.array([best(np.array([j for j in range(nc) if j != i])) for i in range(nc)])
    lo = ah * fac[:, None]
    r = float(np.corrcoef(ah.ravel(), at.ravel())[0, 1])
    return {"r": r,
            "raw_skill": 1.0 - float(np.abs(ah - at).mean()) / signal,
            "loco_skill": 1.0 - float(np.abs(lo - at).mean()) / signal,
            "oracle_skill": 1.0 - float(np.abs(best(np.arange(nc)) * ah - at).mean())
            / signal,
            "loco_factors": fac.tolist(),
            "dispersion": float(np.abs(ah).mean() / signal)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="_v3")
    ap.add_argument("--era", type=int, default=K.METERED_ERA_YEAR0)
    args = ap.parse_args()

    drv = importlib.import_module("11_kansas_run")
    a = drv.assemble(pool=True)
    years = np.arange(K.YEAR0, K.YEAR1 + 1)
    area = a["irr_area"]
    P = K.precipitation()
    deficit = P.mean(axis=1, keepdims=True) - P
    points = {"FLAT": area * R.PRIOR_DEPTH_M,
              "WATERBAL": area * (R.PRIOR_DEPTH_M + 0.5 * deficit / 1000.0),
              "OPENLOOP": a["et_obs"] / 0.80}
    post = {k: np.load(RES / f"kansas_posterior_{k}{args.tag}.npz")["ens"]
            for k in ("ETH", "ET", "H")}
    ens = post["ETH"]

    q_v3, _ = K.metered_annual()
    q_by, share, meta = K.reported_annual()
    q_v2 = q_by.sum(axis=0)
    era = years >= args.era

    res = {"_meta": {"tag": args.tag, "era_year0": int(args.era),
                     "ne_converged": int(ens.shape[0]), "counties": K.COUNTIES,
                     "years": [int(years[0]), int(years[-1])]}}

    # ---------------------------------------------------------------- the audit
    tot = q_v2.sum(axis=0)
    res["audit"] = {
        "note": ("share of the block's reported irrigation volume whose WIMAS report "
                 "carries a meter code (A, M, I), by year; the rest is code G, hours "
                 "of pump operation times a rate"),
        "metered_share_by_year": {int(y): float(q_by[0, :, i].sum() / tot[i])
                                  for i, y in enumerate(years)},
        "min_county_metered_share_by_year": {int(y): float(np.nanmin(share[:, i]))
                                             for i, y in enumerate(years)},
        "metered_share_by_county_2000_2008": {
            c: float(q_by[0, ci, ~era].sum() / q_v2[ci, ~era].sum())
            for ci, c in enumerate(K.COUNTIES)},
        "metered_share_by_county_era": {
            c: float(q_by[0, ci, era].sum() / q_v2[ci, era].sum())
            for ci, c in enumerate(K.COUNTIES)},
        "reported_over_published_truth_by_year": {
            int(y): float(tot[i] / q_v3[:, i].sum()) for i, y in enumerate(years)},
        "reported_over_published_truth_by_county": {
            c: float(q_v2[ci].sum() / q_v3[ci].sum()) for ci, c in enumerate(K.COUNTIES)},
        "n_reports_metered_era": int(np.array(meta["n_reports"])[0][:, era].sum()),
    }
    first_full = next(int(y) for i, y in enumerate(years)
                      if q_by[0, :, i:].sum() / tot[i:].sum() > 0.98
                      and (q_by[0, :, i:].sum(axis=0) / tot[i:] > 0.98).all())
    res["audit"]["first_year_every_later_year_above_98pct"] = first_full

    # ---------------------------------------------------------------- the levels
    res["level"] = {
        "published_truth_all_years": level(ens, points, q_v3),
        "reported_truth_all_years": level(ens, points, q_v2),
        "reported_truth_metered_era": level(
            ens[..., era], {k: v[:, era] for k, v in points.items()}, q_v2[:, era]),
    }
    for row in ("ET", "H"):
        res["level"]["reported_truth_metered_era"][row] = {
            **MT.point_scores(post[row][..., era].mean(axis=0), q_v2[:, era]),
            **MT.coverage(post[row][..., era], q_v2[:, era])}

    # ---------------------------------------------------------------- interannual
    res["anomaly_metered_era"] = {
        "CLOSURE": loco_skill(ens[..., era].mean(axis=0), q_v2[:, era]),
        "ET": loco_skill(post["ET"][..., era].mean(axis=0), q_v2[:, era]),
        "H": loco_skill(post["H"][..., era].mean(axis=0), q_v2[:, era]),
        "OPENLOOP": loco_skill(points["OPENLOOP"][:, era], q_v2[:, era]),
    }
    res["anomaly_published"] = {
        "CLOSURE": loco_skill(ens.mean(axis=0), q_v3),
    }

    # ---------------------------------------------------------------- the changes
    ye = years[era]
    pe = {k: v[:, era] for k, v in points.items()}
    ee = ens[..., era]
    res["contrasts_metered_era"] = {}
    for key, why, aw, bw in CONTRASTS:
        t = float(contrast(q_v2, years, aw, bw))
        e = contrast(ens, years, aw, bw)
        lo, hi = float(np.percentile(e, 5)), float(np.percentile(e, 95))
        row = {"why": why, "baseline_years": list(aw), "period_years": list(bw),
               "metered_pct": t,
               "CLOSURE": {"pct": float(e.mean()), "sd": float(e.std()),
                           "ci90": [lo, hi], "contains_truth": bool(lo <= t <= hi),
                           "excludes_zero": bool(not (lo <= 0.0 <= hi)),
                           "abs_error_pts": abs(float(e.mean()) - t)}}
        for k, v in points.items():
            p = float(contrast(v, years, aw, bw))
            row[k] = {"pct": p, "abs_error_pts": abs(p - t)}
        res["contrasts_metered_era"][key] = row

    s_era, err_era, t_era, e_era = sweep(ee, pe, q_v2[:, era], ye, W)
    s_full2, err_full2, t_full, e_full = sweep(ens, points, q_v2, years, W)
    s_full3, _, _, _ = sweep(ens, points, q_v3, years, W)
    res["sweep_5yr"] = {
        "metered_era_reported_truth": s_era,
        "all_years_reported_truth": s_full2,
        "all_years_published_truth": s_full3,
    }

    curve = {}
    for w in range(2, 9):
        if len(ye) < 2 * w:
            continue
        s, _, _, _ = sweep(ee, pe, q_v2[:, era], ye, w)
        curve[str(w)] = s
    cross = next((int(w) for w in sorted(curve, key=int)
                  if curve[w]["mean_abs_error_pts"]["CLOSURE"]
                  < min(curve[w]["mean_abs_error_pts"][k] for k in points)), None)
    res["window_curve_metered_era"] = {"by_window_years": curve,
                                       "crossover_window_years": cross}

    # Does the closure's error depend on how metered the truth was? Every five-year pair
    # on the full record, grouped by the smaller of its two windows' metered shares.
    sh_year = np.array([q_by[0, :, i].sum() / tot[i] for i in range(len(years))])
    grp = {"both_windows_metered": [], "mixed": [], "both_windows_pre_meter": []}
    for (i, j), ec, ef in zip(pairs_of(len(years), W), err_full2["CLOSURE"],
                              err_full2["OPENLOOP"]):
        sa, sb = sh_year[i:i + W].mean(), sh_year[j:j + W].mean()
        g = ("both_windows_metered" if min(sa, sb) > 0.98
             else "both_windows_pre_meter" if max(sa, sb) < 0.90 else "mixed")
        grp[g].append((ec, ef))
    res["closure_error_by_metered_share"] = {
        g: {"n_pairs": len(v),
            "closure_mean_abs_error_pts": float(np.mean([x for x, _ in v])) if v else None,
            "openloop_mean_abs_error_pts": float(np.mean([x for _, x in v])) if v else None}
        for g, v in grp.items()}

    e = contrast(ens, years, (2013, 2017), (2018, 2022))
    res["resolution"] = {"posterior_sd_pts": float(e.std()),
                         "one_sided_90pct_detectable_pct": 1.2816 * float(e.std())}

    out = RES / f"metered_era{args.tag}.json"
    out.write_text(json.dumps(res, indent=2))

    # ---------------------------------------------------------------- the figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        yr = years
        m = q_by[0].sum(axis=0) / 1e6
        h = q_by[1].sum(axis=0) / 1e6
        ax[0].bar(yr, m, color="#2b6ca3", label="meter-coded (A, M, I)")
        ax[0].bar(yr, h, bottom=m, color="#d9a441", label="hours x rate (G)")
        ax[0].axvline(args.era - 0.5, color="k", lw=1, ls="--")
        ax[0].text(args.era + 5.5, ax[0].get_ylim()[1] * 0.97, "metered era",
                   va="top", ha="center", fontsize=9,
                   bbox=dict(facecolor="white", edgecolor="none", pad=2))
        ax[0].set_ylabel("reported irrigation pumping, Mm3/yr")
        ax[0].set_title("What the WIMAS record is made of", fontsize=10)
        ax[0].legend(fontsize=8, loc="lower left")

        keys = list(ACCOUNTS)
        x = np.arange(len(keys))
        v_full = [s_full3["mean_abs_error_pts"][k] for k in keys]
        v_era = [s_era["mean_abs_error_pts"][k] for k in keys]
        ax[1].bar(x - 0.2, v_full, 0.4, color="#9fb7c9",
                  label=f"2000-2024, published truth ({s_full3['n_pairs']} pairs)")
        ax[1].bar(x + 0.2, v_era, 0.4, color="#2b6ca3",
                  label=f"{args.era}-2024, metered only ({s_era['n_pairs']} pairs)")
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(["closure", "area x depth", "+ weather", "open loop"],
                              fontsize=9)
        ax[1].set_ylabel("mean |error| on a five-year change, points")
        ax[1].set_title("Five-year change, both scorings", fontsize=10)
        ax[1].legend(fontsize=8)

        ws = sorted(curve, key=int)
        for k, c in zip(keys, ("#2b6ca3", "#888888", "#d9a441", "#b04a3c")):
            ax[2].plot([int(w) for w in ws],
                       [curve[w]["mean_abs_error_pts"][k] for w in ws],
                       marker="o", color=c, label=k.lower())
        ax[2].set_xlabel("averaging window, years")
        ax[2].set_ylabel("mean |error| on the change, points")
        ax[2].set_title(f"Window curve, metered era {args.era}-2024", fontsize=10)
        ax[2].legend(fontsize=8)
        fig.tight_layout()
        FIG.mkdir(exist_ok=True)
        # Only the published tag writes the figure the documents embed.
        fig.savefig(FIG / ("fig16_metered_era.png" if args.tag == "_v3"
                           else f"fig16_metered_era{args.tag}.png"), dpi=160)
    except ImportError:
        pass

    # ---------------------------------------------------------------- the report
    au = res["audit"]
    print("WIMAS measurement codes, share of reported volume that is meter-coded:")
    for y in years:
        print("  {}  block {:5.1f}%   lowest county {:5.1f}%   reported/published {:.3f}"
              .format(y, 100 * au["metered_share_by_year"][int(y)],
                      100 * au["min_county_metered_share_by_year"][int(y)],
                      au["reported_over_published_truth_by_year"][int(y)]))
    print("  first year from which every later year is above 98%: {}"
          .format(au["first_year_every_later_year_above_98pct"]))

    print("\nlevel, district-year, MAE Mm3/yr / MAPE % / basin bias % / 90% coverage:")
    for name, L in res["level"].items():
        print("  " + name)
        for k in ("CLOSURE", "ET", "H", "FLAT", "WATERBAL", "OPENLOOP"):
            if k not in L:
                continue
            v = L[k]
            print("    {:<9s} {:6.2f} {:6.1f} {:+6.1f} {}".format(
                k, v["mae_mcm"], v["mape_pct"], v["basin_bias_pct"],
                "{:.2f}".format(v["cover_90"]) if "cover_90" in v else ""))

    print("\ninterannual anomaly, metered era, skill against flat (r / raw / LOCO / oracle):")
    for k, v in res["anomaly_metered_era"].items():
        print("  {:<9s} {:5.2f} {:6.2f} {:6.2f} {:6.2f}".format(
            k, v["r"], v["raw_skill"], v["loco_skill"], v["oracle_skill"]))

    print("\nnamed contrasts inside the metered era:")
    for key, row in res["contrasts_metered_era"].items():
        print("  {:<5s} metered {:+6.1f}%  closure {:+6.1f}% [{:+.1f}, {:+.1f}]  "
              "flat {:+6.1f}%  waterbal {:+6.1f}%  openloop {:+6.1f}%".format(
                  key, row["metered_pct"], row["CLOSURE"]["pct"], *row["CLOSURE"]["ci90"],
                  row["FLAT"]["pct"], row["WATERBAL"]["pct"], row["OPENLOOP"]["pct"]))

    for name, s in res["sweep_5yr"].items():
        print("\nfive-year window pairs, {} ({} pairs):".format(name, s["n_pairs"]))
        for k in ACCOUNTS:
            print("  {:<9s} mean |error| {:5.1f}   median {:5.1f}".format(
                k, s["mean_abs_error_pts"][k], s["median_abs_error_pts"][k]))
        print("  closure closer than FLAT {:.0f}%, WATERBAL {:.0f}%, OPENLOOP {:.0f}%; "
              "coverage {:.0f}%; declares {} with sign right {}".format(
                  s["closure_beats_pct"]["FLAT"], s["closure_beats_pct"]["WATERBAL"],
                  s["closure_beats_pct"]["OPENLOOP"], 100 * s["coverage_90"],
                  s["n_declared_change"], s["n_declared_change_sign_correct"]))

    print("\nwindow curve, metered era:")
    print("  {:>7s} {:>6s} {:>9s} {:>9s} {:>9s} {:>9s} {:>6s}".format(
        "window", "pairs", "CLOSURE", "FLAT", "WATERBAL", "OPENLOOP", "cover"))
    for w in sorted(curve, key=int):
        r = curve[w]
        mm = r["mean_abs_error_pts"]
        print("  {:>5s} yr {:6d} {:9.1f} {:9.1f} {:9.1f} {:9.1f} {:5.0f}%".format(
            w, r["n_pairs"], mm["CLOSURE"], mm["FLAT"], mm["WATERBAL"], mm["OPENLOOP"],
            100 * r["coverage_90"]))
    print("  crossover: {}".format(cross))

    print("\nclosure error by how metered the pair's truth was (full record, 5-yr pairs):")
    for g, v in res["closure_error_by_metered_share"].items():
        print("  {:<24s} n={:3d}  closure {:5.1f}  open loop {:5.1f}".format(
            g, v["n_pairs"], v["closure_mean_abs_error_pts"] or float("nan"),
            v["openloop_mean_abs_error_pts"] or float("nan")))
    print("\nwrote " + str(out.relative_to(ROOT)))


if __name__ == "__main__":
    main()
