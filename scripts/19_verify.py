"""What a regulator actually asks: did abstraction change, by how much, and how sure?

A water account is usually judged on its level. A regulator with a reduction target is
not asking for a level. Saudi Arabia has published a target of a 90 per cent reduction in
non-renewable groundwater use; Kansas writes its Local Enhanced Management Areas as a
percentage cut against a stated baseline period. Both are questions about a **change
between two multi-year periods**, and both need an interval, because a point estimate
cannot say whether a reduction is real.

This script scores every account on that question, against the same withheld meters.

The result separates the accounts by mechanism. Mapped area times a published depth,
and the same corrected for the year's precipitation deficit, are weather models: they
carry the year-to-year wiggle well and they are blind to any change that is not weather.
The closure carries the aquifer's storage, which is the only observation that records
water that actually left the ground, and it is the only account that states an interval.

Three contrasts are scored, all fixed by the public policy record and none chosen from a
score: the Sheridan-6 LEMA period, the district-wide GMD4 LEMA period, and the long
contrast across the record.

    python scripts/19_verify.py --tag _v4

Writes results/verify{tag}.json.
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

# Fixed by the Kansas Department of Agriculture record, not by any score.
# SD-6 is the first Local Enhanced Management Area, Sheridan County, effective 2013.
# The district-wide GMD4 LEMA covers all six counties and ran 2018 to 2022.
CONTRASTS = [
    ("SD6", "the Sheridan-6 LEMA period against its own stated baseline",
     (2005, 2012), (2013, 2017)),
    ("GMD4", "the district-wide GMD4 LEMA period against the five years before it",
     (2013, 2017), (2018, 2022)),
    ("LONG", "the last seven years of the record against the first eight",
     (2000, 2007), (2018, 2024)),
]


def contrast(v: np.ndarray, years: np.ndarray, a: tuple, b: tuple) -> np.ndarray:
    """Percentage change in basin abstraction between two multi-year periods.

    Trailing axis is the year, the one before it the county. Any leading axis is carried
    through, so the same call returns a number for a point estimate and an ensemble of
    numbers for a posterior.
    """
    A = (years >= a[0]) & (years <= a[1])
    B = (years >= b[0]) & (years <= b[1])
    return 100.0 * (v[..., B].mean(-1).sum(-1) / v[..., A].mean(-1).sum(-1) - 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="_v4")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    drv = importlib.import_module("11_kansas_run")
    a = drv.assemble(pool=True)
    q_true, meta = K.metered_annual()
    years = np.arange(K.YEAR0, K.YEAR1 + 1)
    area = a["irr_area"]
    P = K.precipitation()
    deficit = P.mean(axis=1, keepdims=True) - P
    post = np.load(RES / ("kansas_posterior_ETH" + args.tag + ".npz"))
    ens = post["ens"]

    points = {
        "FLAT": ("mapped irrigated area x one acre-foot per acre",
                 area * R.PRIOR_DEPTH_M),
        "WATERBAL": ("the same, plus half the year's precipitation deficit",
                     area * (R.PRIOR_DEPTH_M + 0.5 * deficit / 1000.0)),
        "OPENLOOP": ("unmixed evapotranspiration over a fixed efficiency of 0.80",
                     a["et_obs"] / 0.80),
    }

    res = {"_meta": {"tag": args.tag, "ne_converged": int(ens.shape[0]),
                     "counties": K.COUNTIES, "years": [K.YEAR0, K.YEAR1], **meta}}
    err = {k: [] for k in list(points) + ["CLOSURE"]}
    for key, why, aw, bw in CONTRASTS:
        truth = float(contrast(q_true, years, aw, bw))
        e = contrast(ens, years, aw, bw)                    # (ne,)
        lo, hi = float(np.percentile(e, 5)), float(np.percentile(e, 95))
        row = {"why": why, "baseline_years": list(aw), "period_years": list(bw),
               "metered_pct": truth,
               "CLOSURE": {"pct": float(e.mean()), "sd": float(e.std()),
                           "ci90": [lo, hi], "contains_truth": bool(lo <= truth <= hi),
                           "excludes_zero": bool(not (lo <= 0.0 <= hi)),
                           "abs_error_pts": abs(float(e.mean()) - truth)}}
        err["CLOSURE"].append(abs(float(e.mean()) - truth))
        for k, (lab, v) in points.items():
            p = float(contrast(np.asarray(v, dtype=float), years, aw, bw))
            row[k] = {"pct": p, "interval": None, "abs_error_pts": abs(p - truth),
                      "label": lab}
            err[k].append(abs(p - truth))
        res[key] = row

    res["_summary"] = {
        "mean_abs_error_pts": {k: float(np.mean(v)) for k, v in err.items()},
        "only_account_with_an_interval": "CLOSURE",
    }

    # Three contrasts are three contrasts. The same question is asked of every pair of
    # non-overlapping five-year windows the record admits, so the ranking is not carried
    # by the three the policy record happens to name.
    W = 5
    starts = range(0, len(years) - W + 1)
    pairs = [(i, j) for i in starts for j in starts if j >= i + W]
    acc = {k: [] for k in list(points) + ["CLOSURE"]}
    cover = declared = declared_right = 0
    truth_pct, est_pct, flag = [], [], []
    for i, j in pairs:
        A = np.zeros(len(years), dtype=bool); A[i:i + W] = True
        B = np.zeros(len(years), dtype=bool); B[j:j + W] = True
        aw = (years[A][0], years[A][-1]); bw = (years[B][0], years[B][-1])
        t = float(contrast(q_true, years, aw, bw))
        e = contrast(ens, years, aw, bw)
        acc["CLOSURE"].append(abs(float(e.mean()) - t))
        lo, hi = np.percentile(e, 5), np.percentile(e, 95)
        cover += int(lo <= t <= hi)
        if not (lo <= 0.0 <= hi):
            declared += 1
            declared_right += int(np.sign(e.mean()) == np.sign(t))
        truth_pct.append(t)
        est_pct.append(float(e.mean()))
        flag.append(not (lo <= 0.0 <= hi))
        for k, (lab, v) in points.items():
            acc[k].append(abs(float(contrast(np.asarray(v, dtype=float), years, aw, bw)) - t))
    # The window length is not a free choice, it is the physics. A weather model carries
    # the high-frequency half of the signal and saturates; the aquifer integrates storage
    # and keeps improving as the window lengthens. Where the two curves cross is the
    # shortest averaging period over which closing the loop is worth doing, and that
    # number is a design rule for a monitoring programme.
    curve = {}
    for w in range(2, 9):
        st = range(0, len(years) - w + 1)
        pp = [(i, j) for i in st for j in st if j >= i + w]
        e = {k: [] for k in list(points) + ["CLOSURE"]}
        cv = dc = dr = 0
        for i, j in pp:
            A = np.zeros(len(years), dtype=bool); A[i:i + w] = True
            B = np.zeros(len(years), dtype=bool); B[j:j + w] = True
            aw2 = (years[A][0], years[A][-1]); bw2 = (years[B][0], years[B][-1])
            t = float(contrast(q_true, years, aw2, bw2))
            en = contrast(ens, years, aw2, bw2)
            e["CLOSURE"].append(abs(float(en.mean()) - t))
            lo2, hi2 = np.percentile(en, 5), np.percentile(en, 95)
            cv += int(lo2 <= t <= hi2)
            if not (lo2 <= 0.0 <= hi2):
                dc += 1
                dr += int(np.sign(en.mean()) == np.sign(t))
            for k, (lab, v) in points.items():
                e[k].append(abs(float(contrast(np.asarray(v, dtype=float), years,
                                               aw2, bw2)) - t))
        curve[str(w)] = {
            "n_pairs": len(pp),
            "mean_abs_error_pts": {k: float(np.mean(v)) for k, v in e.items()},
            "coverage_90": float(cv / len(pp)),
            "n_declared_change": dc, "n_declared_change_sign_correct": dr,
        }
    # The crossover is the shortest window at which the closure beats *every* meter-free
    # bar, not the one that happens to be strongest at a chosen window. Below it, a
    # weather model is the better instrument and the entry should say so.
    def best_bar_at(w):
        return min(points, key=lambda k: curve[w]["mean_abs_error_pts"][k])
    cross = next((int(w) for w in sorted(curve, key=int)
                  if curve[w]["mean_abs_error_pts"]["CLOSURE"]
                  < min(curve[w]["mean_abs_error_pts"][k] for k in points)), None)
    res["_window_curve"] = {
        "by_window_years": curve,
        "strongest_meter_free_bar_by_window": {w: best_bar_at(w) for w in curve},
        "crossover_window_years": cross,
        "note": ("shortest window at which the closure beats every meter-free bar; "
                 "below it a weather model is the better instrument"),
    }

    n = len(pairs)
    truth_pct = np.array(truth_pct); est_pct = np.array(est_pct); flag = np.array(flag)

    # The direction of the change is not a test on this record and must not be reported as
    # one. Abstraction fell over almost every window pair, so an estimator that says
    # "down" every time scores whatever the closure scores on sign. What is a test is
    # whether declaring a change tracks the size of the real one, which no constant
    # satisfies. Reported as the area under the curve of the declaration against the
    # metered magnitude, with a permutation null.
    big, small = np.abs(truth_pct)[flag], np.abs(truth_pct)[~flag]
    auc = float((big[:, None] > small[None, :]).mean()
                + 0.5 * (big[:, None] == small[None, :]).mean())
    rng = np.random.default_rng(0)
    null = np.empty(20000)
    for b in range(null.size):
        pm = rng.permutation(flag)
        u, v = np.abs(truth_pct)[pm], np.abs(truth_pct)[~pm]
        null[b] = ((u[:, None] > v[None, :]).mean()
                   + 0.5 * (u[:, None] == v[None, :]).mean())
    detect = {
        "n_metered_change_negative": int((truth_pct < 0).sum()),
        "always_down_scores_on_declared": float(max((truth_pct[flag] < 0).mean(),
                                                    (truth_pct[flag] > 0).mean())),
        "sign_is_not_a_test_on_this_record": True,
        "declaration_auc_vs_metered_magnitude": auc,
        "declaration_auc_permutation_p": float((null >= auc).mean()),
        "metered_pct_where_declared": float(big.mean()),
        "metered_pct_where_not_declared": float(small.mean()),
        "change_r": float(np.corrcoef(est_pct, truth_pct)[0, 1]),
        "amplitude_slope": float(np.polyfit(est_pct, truth_pct, 1)[0]),
    }
    res["_sweep"] = {
        "window_years": W, "n_pairs": n,
        "mean_abs_error_pts": {k: float(np.mean(v)) for k, v in acc.items()},
        "median_abs_error_pts": {k: float(np.median(v)) for k, v in acc.items()},
        "closure_beats_pct": {k: float(100.0 * np.mean(np.array(acc["CLOSURE"])
                                                       < np.array(acc[k])))
                              for k in points},
        "coverage_90": float(cover / n),
        "n_declared_change": declared,
        "n_declared_change_sign_correct": declared_right,
        **detect,
    }
    # The verification resolution: the posterior spread on a five-year-against-five-year
    # basin contrast is what decides the smallest reduction this observing system can
    # separate from no change.
    e = contrast(ens, years, (2013, 2017), (2018, 2022))
    sd = float(e.std())
    res["_resolution"] = {
        "posterior_sd_pts": sd,
        "one_sided_90pct_detectable_pct": 1.2816 * sd,
        "two_sided_90pct_detectable_pct": 1.6449 * sd,
        "note": ("smallest basin-wide five-year change this observing system separates "
                 "from no change, at the stated confidence"),
    }
    # The Sheridan-6 LEMA at county resolution, which is where the instrument runs out.
    # SD-6 covers 256 km2 inside a 2,331 km2 county, so a 30 per cent cut inside it is
    # about a 9 per cent cut on the county. Decatur is excluded from the control group
    # because its mapped irrigated area halves over the window, which is a map artefact
    # and not a change in use. This is reported because it sizes the claim.
    pre = (years >= 2005) & (years <= 2012)
    post_w = (years >= 2013) & (years <= 2017)
    treated = K.COUNTIES.index("SD")
    control = [i for i, c in enumerate(K.COUNTIES) if c not in ("SD", "DC")]

    def did(vol):
        d = np.asarray(vol, dtype=float) / area * 1000.0        # applied depth, mm
        t = 100.0 * (d[..., treated, :][..., post_w].mean(-1)
                     / d[..., treated, :][..., pre].mean(-1) - 1.0)
        c = 100.0 * (d[..., control, :][..., post_w].mean((-2, -1))
                     / d[..., control, :][..., pre].mean((-2, -1)) - 1.0)
        return t, c, t - c

    t_m, c_m, d_m = did(q_true)
    t_e, c_e, d_e = did(ens)
    res["_sd6_county_did"] = {
        "why": ("the SD-6 LEMA is 256 km2 inside a 2,331 km2 county; this is the "
                "resolution limit of the instrument, and it is a negative result"),
        "treated": "SD", "control": [K.COUNTIES[i] for i in control],
        "control_excludes": ["DC"],
        "metered_treated_pct": float(t_m), "metered_control_pct": float(c_m),
        "metered_did_pts": float(d_m),
        "closure_did_pts": float(d_e.mean()), "closure_did_sd": float(d_e.std()),
        "closure_did_ci90": [float(np.percentile(d_e, 5)),
                             float(np.percentile(d_e, 95))],
        "sign_agrees": bool(np.sign(d_e.mean()) == np.sign(d_m)),
    }

    out = args.out or ("verify" + args.tag + ".json")
    (RES / out).write_text(json.dumps(res, indent=2))

    print("{:<6s} {:>9s} {:>19s} {:>9s} {:>9s} {:>9s}".format(
        "", "METERED", "CLOSURE", "FLAT", "WATERBAL", "OPENLOOP"))
    for key, why, aw, bw in CONTRASTS:
        r = res[key]
        print("{:<6s} {:+8.1f}% {:+8.1f}% +-{:4.1f} {:+8.1f}% {:+8.1f}% {:+8.1f}%".format(
            key, r["metered_pct"], r["CLOSURE"]["pct"], r["CLOSURE"]["sd"],
            r["FLAT"]["pct"], r["WATERBAL"]["pct"], r["OPENLOOP"]["pct"]))
    s = res["_summary"]["mean_abs_error_pts"]
    print("\nmean absolute error on the three contrasts, percentage points:")
    for k in ("CLOSURE", "FLAT", "WATERBAL", "OPENLOOP"):
        print("  {:<9s} {:5.1f}".format(k, s[k]))
    print("\nverification resolution: posterior sd {:.1f} pts, so a basin-wide five-year "
          "change\nis separated from zero at 90 per cent one-sided confidence when it "
          "exceeds {:.1f} per cent".format(sd, 1.2816 * sd))

    s = res["_sweep"]
    print("\nevery non-overlapping five-year window pair in the record, {} of them:"
          .format(s["n_pairs"]))
    for k in ("CLOSURE", "FLAT", "WATERBAL", "OPENLOOP"):
        print("  {:<9s} mean |error| {:5.1f} pts   median {:5.1f}".format(
            k, s["mean_abs_error_pts"][k], s["median_abs_error_pts"][k]))
    print("  the closure is closer than FLAT on {:.0f}%, WATERBAL {:.0f}%, OPENLOOP {:.0f}%"
          .format(*[s["closure_beats_pct"][k] for k in ("FLAT", "WATERBAL", "OPENLOOP")]))
    print("  its 90 per cent interval covers the metered contrast in {:.0f} per cent of pairs"
          .format(100 * s["coverage_90"]))
    print("  it declares a change in {} pairs; the metered change there averages {:.1f} "
          "per cent".format(s["n_declared_change"], s["metered_pct_where_declared"]))
    print("  against {:.1f} where it declares none, AUC {:.3f}, permutation p {:.4f}"
          .format(s["metered_pct_where_not_declared"],
                  s["declaration_auc_vs_metered_magnitude"],
                  s["declaration_auc_permutation_p"]))
    print("  direction is NOT a test here: the metered change is negative in {} of {} "
          "pairs,".format(s["n_metered_change_negative"], s["n_pairs"]))
    print("  so an estimator that always says down scores {:.0f} per cent on the declared set"
          .format(100 * s["always_down_scores_on_declared"]))

    z = res["_sd6_county_did"]
    print("")
    print("Sheridan-6 LEMA at county resolution, the resolution limit:")
    print("  metered difference in differences {:+.1f} points; closure {:+.1f} +- {:.1f}, "
          "90 per cent [{:+.1f}, {:+.1f}]"
          .format(z["metered_did_pts"], z["closure_did_pts"], z["closure_did_sd"],
                  *z["closure_did_ci90"]))
    print("  sign agrees: {}. A policy on a tenth of a county is below what this "
          "observing system resolves.".format(z["sign_agrees"]))

    c = res["_window_curve"]
    print("\nmean absolute error on the change against the length of the window, points:")
    print("  {:>7s} {:>6s} {:>9s} {:>9s} {:>9s} {:>9s} {:>6s}".format(
        "window", "pairs", "CLOSURE", "FLAT", "WATERBAL", "OPENLOOP", "cover"))
    for w in sorted(c["by_window_years"], key=int):
        r = c["by_window_years"][w]
        m = r["mean_abs_error_pts"]
        print("  {:>5s} yr {:6d} {:9.1f} {:9.1f} {:9.1f} {:9.1f} {:5.0f}%".format(
            w, r["n_pairs"], m["CLOSURE"], m["FLAT"], m["WATERBAL"], m["OPENLOOP"],
            100 * r["coverage_90"]))
    print("  the closure beats every meter-free bar from a {}-year window upward"
          .format(c["crossover_window_years"]))
    print("  below that the best of them is a weather model, and it is the better "
          "instrument")
    print("\nwrote results/" + out)


if __name__ == "__main__":
    main()
