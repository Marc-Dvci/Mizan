"""What the external mass trend prior costs, and what the target basin says it should be.

The gravity operator reads

    G(t) = alpha * dS(t) / A + external(t) + eps

Script 23 puts the mascon gain `alpha` on an axis. This one does the same for the other
half of the pair, `external`, whose linear term the entry constrains to plus or minus
1.0 mm/yr on a physical argument: in a hyper-arid basin with no surface water and no
snow, the trend in total water storage is the trend in groundwater.

That argument is checkable on the target basin, and L3 already checks it. The four
unirrigated control boxes it reads over the Arabian shield and the Rub' al Khali carry
gravimetric trends of several mm/yr in their own right. If a signal of that size is
present over the Saq and is not local groundwater, a prior of plus or minus 1.0 mm/yr is
far too tight, and the absolute scale of the account would inherit the difference.

So the row is repeated with the trend prior widened to the scale the controls measure,
and with nothing else changed:

    make drift        # the two runs
    python scripts/25_drift.py

Writes results/drift.json and figures/fig15_drift.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mizan import estimator as E, figures as F, truth as T

RES = ROOT / "results"
FIG = ROOT / "figures"

# (results file, run tag, label). The shipped configuration is the 1.0 mm/yr row, which
# is the same file the gain sweep reads at its published gain width: one run, two axes.
SWEEP = [
    ("gain_pub.json", "_gpub", "the shipped constraint"),
    ("drift_wide.json", "_dwide", "widened to the control scale"),
    ("drift_control.json", "_dctl", "widened past the largest control"),
]


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def nuisance(tag: str) -> dict:
    """The posterior gain and external trend of one run, against withheld truth."""
    p = RES / f"posterior_H{tag}.npz"
    if not p.exists():
        return {}
    z = np.load(p)
    X, ok = z["X"], z["ok"] if "ok" in z.files else np.ones(z["X"].shape[1], bool)
    a = np.asarray(X[E.LAYOUT["grace_alpha"]][:, ok]).ravel()
    d = np.asarray(X[E.LAYOUT["grace_drift"]][:, ok])[0]
    return {"alpha_hat": float(a.mean()), "alpha_err": float(a.mean() - T.GRACE_ALPHA),
            "drift_trend_hat": float(d.mean()),
            "drift_trend_sd_post": float(d.std()),
            "drift_trend_err": float(d.mean() - float(T.GRACE_DRIFT[0]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="drift.json")
    ap.add_argument("--fig", type=str, default="fig15_drift.png")
    args = ap.parse_args()

    al = load("aljawf.json")
    ctl = al["grace"]["controls"] if al else {}
    # A mascon trend in cm/decade is mm/yr, so the controls are read straight across.
    controls = sorted((abs(v["cm_decade"]), k) for k, v in ctl.items())
    saq = abs(al["grace"]["saq_cm_decade"]) if al else float("nan")

    res = {"_operator": "G(t) = alpha * dS(t) / A + external(t) + eps",
           "_shipped_prior_mm_yr": 1.0,
           "_withheld_truth": {"external_trend_mm_yr": float(T.GRACE_DRIFT[0]),
                               "alpha": float(T.GRACE_ALPHA)},
           "_l3_controls_mm_yr": {k: round(v, 3) for v, k in controls},
           "_l3_control_range_mm_yr": [controls[0][0], controls[-1][0]] if controls else None,
           "_l3_saq_trend_mm_yr": saq}

    rows = []
    for name, tag, label in SWEEP:
        d = load(name)
        if not d or "H" not in d:
            continue
        h, m = d["H"], d["_meta"]
        r = {"file": name, "label": label,
             "drift_trend_prior_sd": m.get("drift_trend_sd"),
             "alpha_prior_sd": m.get("alpha_sd"),
             "mae_mcm": h["mae_mcm"], "mape_pct": h["mape_pct"],
             "basin_bias_pct": h["basin_bias_pct"],
             "cover_90": h.get("cover_90"), "width90_mcm": h.get("width90_mcm"),
             "crps_mcm": h.get("crps_mcm")}
        r.update(nuisance(tag))
        rows.append(r)
    res["sweep"] = rows

    if len(rows) >= 2:
        base, wide = rows[0], rows[-1]
        res["_verdict"] = {
            "prior_width_ratio": (wide["drift_trend_prior_sd"]
                                  / base["drift_trend_prior_sd"]),
            "bias_shipped_pct": base["basin_bias_pct"],
            "bias_widest_pct": wide["basin_bias_pct"],
            "bias_span_pts": float(max(r["basin_bias_pct"] for r in rows)
                                   - min(r["basin_bias_pct"] for r in rows)),
            "mae_shipped_mcm": base["mae_mcm"],
            "mae_widest_mcm": wide["mae_mcm"],
            "mae_ratio": float(wide["mae_mcm"] / base["mae_mcm"]),
            "width90_ratio": (float(wide["width90_mcm"] / base["width90_mcm"])
                              if base.get("width90_mcm") else None),
            "trend_err_shipped": base.get("drift_trend_err"),
            "trend_err_widest": wide.get("drift_trend_err"),
            "cover_90_widest": wide.get("cover_90"),
            "note": ("the pair is degenerate, not either member of it. With the gain "
                     "held at its shipped prior, widening the external trend prior "
                     "tenfold costs a few per cent of accuracy and leaves the coverage "
                     "unchanged, so the constraint the target basin's controls do not "
                     "support is not the one the answer rests on. Section 5.5 releases "
                     "both together and the estimate loses half its accuracy: it takes "
                     "two free nuisances to lose the scale, and pinning either one is "
                     "enough"),
        }

    # The two nuisances form a 2 by 2: each of them pinned at the shipped prior or
    # released. Three of the four cells exist for other reasons, so the design is
    # assembled here rather than re-run.
    CELLS = [("gain_pub.json", "_gpub", False, False),
             ("drift_control.json", "_dctl", False, True),
             ("gain_alphafree.json", "_gafree", True, False),
             ("gain_free.json", "_gfree", True, True)]
    fact = []
    for name, tag, gfree, dfree in CELLS:
        d = load(name)
        if not d or "H" not in d:
            continue
        h, m = d["H"], d["_meta"]
        cell = {"file": name, "gain_free": gfree, "trend_free": dfree,
                "alpha_prior_sd": m["alpha_sd"], "drift_trend_prior_sd":
                m.get("drift_trend_sd"), "mae_mcm": h["mae_mcm"],
                "basin_bias_pct": h["basin_bias_pct"], "cover_90": h.get("cover_90"),
                "width90_mcm": h.get("width90_mcm")}
        cell.update(nuisance(tag))
        fact.append(cell)
    if len(fact) == 4:
        base = fact[0]["mae_mcm"]
        res["_factorial"] = {
            "cells": fact,
            "mae_ratio_trend_only": fact[1]["mae_mcm"] / base,
            "mae_ratio_gain_only": fact[2]["mae_mcm"] / base,
            "mae_ratio_both": fact[3]["mae_mcm"] / base,
            "note": ("the pair is not symmetric. Released on its own the external "
                     "trend costs a couple of per cent of the district error; released "
                     "on its own, with the trend still held, the gain costs a third; "
                     "released together they cost about half. The gain carries the "
                     "absolute scale and the trend does not, so the gain is the "
                     "constraint that has to be defended, and it is the one that is "
                     "computable on the target basin. In both released-gain cells the "
                     "posterior gain lands within 0.005 of truth while the error is a "
                     "third worse, which is a second demonstration that a "
                     "well-recovered gain is not evidence the leg was read correctly"),
        }
    else:
        res["_factorial"] = {"cells": fact,
                             "note": "the 2 by 2 is incomplete; run `make gain drift`"}

    (RES / args.out).write_text(json.dumps(res, indent=2))

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))

    a0 = ax[0]
    names = [k for _, k in controls]
    vals = [v for v, _ in controls]
    y = np.arange(len(names) + 1)
    a0.barh(y[:-1], vals, color=F.MUTED)
    a0.barh([y[-1]], [saq], color=F.ACCENT)
    lab = [n.split(",")[0] for n in names] + ["the Saq"]
    for k, v in enumerate(vals + [saq]):
        a0.text(v + 0.2, y[k], "{:.1f}".format(v), va="center", fontsize=8)
    a0.axvline(1.0, color=F.WARM, ls=":", lw=1.3)
    a0.annotate("the prior the estimator ships,\nplus or minus 1.0 mm/yr",
                xy=(1.05, 0.5), xytext=(4.6, 0.5), textcoords="data", va="center",
                arrowprops=dict(arrowstyle="->", color=F.WARM, lw=1.0),
                fontsize=8, color=F.WARM, fontweight="bold")
    a0.set_yticks(y)
    a0.set_yticklabels(lab, fontsize=8)
    a0.invert_yaxis()
    a0.set_xlim(0, max(vals + [saq]) * 1.35)
    a0.set_xlabel("magnitude of the gravimetric trend, mm/yr")
    a0.set_title("What the target basin's own controls measure")

    a1 = ax[1]
    if rows:
        x = [r["drift_trend_prior_sd"] for r in rows]
        b = [r["basin_bias_pct"] for r in rows]
        m = [r["mae_mcm"] for r in rows]
        a1.plot(x, b, marker="o", ms=6, lw=2.0, color=F.ACCENT,
                label="basin abstraction bias, left axis")
        for r in rows:
            a1.annotate("{:+.1f}%".format(r["basin_bias_pct"]),
                        (r["drift_trend_prior_sd"], r["basin_bias_pct"]),
                        textcoords="offset points", xytext=(8, -12), fontsize=8,
                        color=F.ACCENT)
        a2 = a1.twinx()
        a2.plot(x, m, marker="s", ms=5, lw=1.6, ls="--", color=F.MUTED,
                label="mean absolute error, right axis")
        a2.set_ylabel("district-annual mean absolute error, Mm3/yr", color=F.MUTED)
        a2.tick_params(axis="y", colors=F.MUTED)
        a1.axhline(0.0, color=F.INK, lw=0.8)
        if controls:
            a1.axvspan(controls[0][0], controls[-1][0], color=F.SAND, alpha=0.20, lw=0)
            a1.annotate("what the L3 controls measure", xy=(controls[0][0], 0.985),
                        xycoords=("data", "axes fraction"), va="top",
                        textcoords="offset points", xytext=(6, 0), fontsize=8,
                        color="#8a7a1f")
        # Both axes are held on the span the gain axis covers, because the point of
        # this panel is the size of the effect and an axis scaled to it would hide that.
        gn = load("gain.json")
        if gn and gn.get("sweep"):
            gb = [r["basin_bias_pct"] for r in gn["sweep"] if not r["free"]]
            gm = [r["mae_mcm"] for r in gn["sweep"] if not r["free"]]
            a1.set_ylim(min(gb) - 1.4, max(gb) + 2.4)
            a2.set_ylim(0.0, max(gm) * 1.14)
            # The same two quantities across the gain axis, as a scale bar. Without it
            # an axis fitted to these three points would make a flat result look steep.
            xr = a1.get_xlim()
            xa = xr[0] + 0.055 * (xr[1] - xr[0])
            a1.annotate("", xy=(xa, min(gb)), xytext=(xa, max(gb)),
                        arrowprops=dict(arrowstyle="<->", color=F.WARM, lw=1.4))
            a1.annotate("what the gain prior of section 5.5\nmoves the same quantity "
                        "across:\n{:+.1f} to {:+.1f} per cent".format(min(gb), max(gb)),
                        xy=(xa, max(gb)), va="bottom",
                        textcoords="offset points", xytext=(9, -4), fontsize=8,
                        color=F.WARM)
        h1, l1 = a1.get_legend_handles_labels()
        h2, l2 = a2.get_legend_handles_labels()
        a1.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8, framealpha=0.95)
    a1.set_xlabel("width of the external trend prior, mm/yr")
    a1.set_ylabel("basin abstraction bias, per cent")
    a1.set_title("Widening it tenfold barely moves the account")

    F.save(fig, FIG / args.fig)

    # ------------------------------------------------------------------ report
    print("The L3 controls, unirrigated boxes, magnitude of trend in mm/yr:")
    for v, k in controls:
        print("  {:50s} {:5.1f}".format(k[:50], v))
    print("  {:50s} {:5.1f}".format("the Saq itself", saq))
    print("\nWhat the external trend prior costs:")
    print("  {:36s} {:>9s} {:>9s} {:>11s} {:>8s}".format(
        "external trend prior", "sd mm/yr", "MAE", "basin bias", "cov90"))
    for r in rows:
        print("  {:36s} {:9.1f} {:9.2f} {:+10.1f}% {:8.2f}".format(
            r["label"], r["drift_trend_prior_sd"], r["mae_mcm"], r["basin_bias_pct"],
            r["cover_90"] if r["cover_90"] is not None else float("nan")))
    print("\n  posterior external trend against a withheld true {:+.1f} mm/yr:".format(
        float(T.GRACE_DRIFT[0])))
    for r in rows:
        if "drift_trend_hat" in r:
            print("    {:36s} {:+.2f} plus or minus {:.2f}".format(
                r["label"], r["drift_trend_hat"], r["drift_trend_sd_post"]))
    fa = res.get("_factorial", {}).get("cells", [])
    if len(fa) == 4:
        print("\nThe two nuisances as a 2 by 2:")
        print("  {:16s} {:16s} {:>9s} {:>11s}".format(
            "gain prior", "trend prior", "MAE", "basin bias"))
        for c in fa:
            print("  {:16s} {:16s} {:9.2f} {:+10.1f}%".format(
                "released" if c["gain_free"] else "shipped",
                "released" if c["trend_free"] else "shipped",
                c["mae_mcm"], c["basin_bias_pct"]))
    print("\nwrote results/{} and figures/{}".format(args.out, args.fig))


if __name__ == "__main__":
    main()
