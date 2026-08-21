"""The bars a reviewer can build without an aquifer, scored on the same meters.

An entry that reports its own score against a weak comparison has reported nothing. This
script builds every meter-free account of Kansas abstraction that can be written down
from public data in a few lines, scores all of them against the same 3,545 withheld
water rights, and scores them on two quantities rather than one:

* the **level**, mean absolute error on county-year volume, which is what a water
  balance is usually judged on;
* the **change**, mean absolute error on the year-over-year difference, which is the
  quantity a regulator with a reduction target is actually asking about.

The two rank the accounts differently, and that is the point of the table.

    python scripts/18_ladder.py --tag _v4

Writes results/ladder{tag}.json.
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


def change_scores(est: np.ndarray, tru: np.ndarray) -> dict:
    """Skill on the year-over-year difference, which no level metric sees.

    Both arguments are m3/yr, as the meters are read; the reported error is Mm3/yr, on the
    same scale as every level score in the project.
    """
    de, dt = np.diff(est, axis=1) / 1e6, np.diff(tru, axis=1) / 1e6
    sd = de.std()
    return {"change_mae_mcm": float(np.abs(de - dt).mean()),
            "change_r": float(np.corrcoef(de.ravel(), dt.ravel())[0, 1]) if sd > 0 else 0.0,
            "change_amplitude_ratio": float(sd / dt.std()),
            "change_skill_vs_flat": float(1.0 - np.abs(de - dt).mean()
                                          / np.abs(dt).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="_v4")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    drv = importlib.import_module("11_kansas_run")
    a = drv.assemble(pool=True)
    q_true, meta = K.metered_annual()
    area = a["irr_area"]
    P = K.precipitation()
    deficit = P.mean(axis=1, keepdims=True) - P
    post = np.load(RES / ("kansas_posterior_ETH" + args.tag + ".npz"))
    closure = post["ens"].mean(axis=0)

    rows = [
        ("ZERO_CHANGE", "each county held at its own record mean, which needs the meters",
         np.repeat(q_true.mean(axis=1, keepdims=True), q_true.shape[1], axis=1), True),
        ("FLAT", "mapped irrigated area x one acre-foot per acre",
         area * R.PRIOR_DEPTH_M, False),
        ("WATERBAL25", "the same, plus a quarter of the year's precipitation deficit",
         area * (R.PRIOR_DEPTH_M + 0.25 * deficit / 1000.0), False),
        ("WATERBAL50", "the same, plus half the year's precipitation deficit",
         area * (R.PRIOR_DEPTH_M + 0.50 * deficit / 1000.0), False),
        ("WATERBAL100", "the same, plus the whole precipitation deficit",
         area * (R.PRIOR_DEPTH_M + 1.00 * deficit / 1000.0), False),
        ("OPENLOOP", "unmixed evapotranspiration over a fixed efficiency of 0.80",
         a["et_obs"] / 0.80, False),
        ("CLOSURE", "the closure, evapotranspiration and heads", closure, False),
    ]

    res = {"_meta": {"tag": args.tag, "counties": K.COUNTIES,
                     "years": [K.YEAR0, K.YEAR1], **meta}}
    for key, label, est, needs_meter in rows:
        sc = MT.point_scores(np.asarray(est, dtype=float), q_true)
        sc.update(change_scores(np.asarray(est, dtype=float), q_true))
        sc["label"] = label
        sc["needs_a_meter"] = needs_meter
        res[key] = sc

    # Where each account's interannual variance comes from. This is what makes the
    # ranking an explanation rather than a lucky metric: an account that is all weather
    # cannot carry a trend, and a trend is what a policy is.
    Pb = P.mean(axis=0)
    Pb = (Pb - Pb.mean()) / Pb.std()
    tt = np.arange(q_true.shape[1], dtype=float)
    tt = (tt - tt.mean()) / tt.std()
    decomp = {}
    for key, est in [("METERED", q_true)] + [(k, v) for k, _, v, _ in rows]:
        b = np.asarray(est, dtype=float).sum(axis=0) / 1e6
        b = b - b.mean()
        var = float((b ** 2).sum())
        if var < 1e-6:
            # An account with no interannual variance has no weather share to report.
            decomp[key] = {"weather_share_pct": None,
                           "trend_after_weather_mcm_per_sd_year": 0.0,
                           "note": "constant in time"}
            continue
        fit = np.polyval(np.polyfit(Pb, b, 1), Pb)
        decomp[key] = {
            "weather_share_pct": float(100 * (1 - ((b - fit) ** 2).sum() / var)),
            "trend_after_weather_mcm_per_sd_year": float(np.polyfit(tt, b - fit, 1)[0]),
        }
    res["_variance_decomposition"] = decomp

    # The closure's error against the precipitation anomaly. A recharge held constant in
    # time forces weather-driven head variation onto pumping, and this is where it shows.
    pa = P - P.mean(axis=1, keepdims=True)
    r = (closure - q_true) / 1e6
    ra = r - r.mean(axis=1, keepdims=True)
    res["_closure_error_vs_precipitation"] = {
        "r": float(np.corrcoef(pa.ravel(), ra.ravel())[0, 1]),
        "note": ("county-year closure error against the county precipitation anomaly; "
                 "a value away from zero means weather is still leaking into the estimate"),
    }

    keys = [k for k in res if not k.startswith("_")]
    res["_ranking"] = {
        "by_level": sorted(keys, key=lambda k: res[k]["mae_mcm"]),
        "by_change": sorted(keys, key=lambda k: res[k]["change_mae_mcm"]),
    }
    out = args.out or ("ladder" + args.tag + ".json")
    (RES / out).write_text(json.dumps(res, indent=2))

    print("{:<13s} {:>7s} {:>7s} {:>8s} {:>7s} {:>6s}".format(
        "account", "LEVEL", "MAPE", "CHANGE", "r", "amp"))
    for k in keys:
        v = res[k]
        print("{:<13s} {:7.2f} {:6.1f}% {:8.2f} {:+7.3f} {:6.2f}  {}".format(
            k, v["mae_mcm"], v["mape_pct"], v["change_mae_mcm"], v["change_r"],
            v["change_amplitude_ratio"], v["label"]))
    print("")
    print("where each account's interannual variance comes from:")
    for k in ["METERED"] + keys:
        d = res["_variance_decomposition"][k]
        w = d["weather_share_pct"]
        print("  {:<13s} weather {:>6s}   trend left after weather {:+7.1f} Mm3/sd-yr"
              .format(k, "n/a" if w is None else "{:.1f}%".format(w),
                      d["trend_after_weather_mcm_per_sd_year"]))
    print("")
    print("closure error against the precipitation anomaly: r = {:+.3f}"
          .format(res["_closure_error_vs_precipitation"]["r"]))
    print("\nby level : " + " < ".join(res["_ranking"]["by_level"]))
    print("by change: " + " < ".join(res["_ranking"]["by_change"]))
    print("\nwrote results/" + out)


if __name__ == "__main__":
    main()
