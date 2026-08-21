"""Is the mascon gain identifiable, and what does its uncertainty cost the answer?

The gravity operator reads

    G(t) = alpha * dS(t) / A + external(t) + eps

and `alpha`, the mascon gain, multiplies the quantity the gravity leg exists to supply.
Basin storage falls close to linearly over the record, so a free `alpha` and a free linear
external trend multiply nearly the same function of time and absorb the absolute scale of
the answer between them. The entry treats `alpha` as computed from the mascon geometry
rather than fitted, which is a prior, and the technical note says every other result rests
on that choice.

This script puts that choice on an axis and scores it. Two questions, both of which a
technical reviewer asks and neither of which the ablation grid answers, because that grid
varies which legs are assimilated and never the gain.

**Is the gain identifiable?** Read as the share of its prior variance the posterior
removes, across the ablation rows that are already on disk. A configuration that returns
the prior untouched has learned nothing about the gain.

**What does a wrong gain cost?** The twin's true gain is 0.78 and the shipped prior is
0.85 plus or minus 0.04, so the prior is wrong by 1.75 of its own standard deviations by
construction. The sweep therefore measures the thing a reviewer actually needs to know,
which is not what a correct gain is worth but what an incorrect one costs, and how much
of that cost the width of the prior buys back.

**What does its uncertainty cost?** Read two ways. The controlled reading is the sweep:
the same row run at a sequence of gain prior widths, so the absolute scale and the
interval width can be plotted against the width of the assumption. The cheap reading is
the covariance between the gain and the basin total inside one posterior ensemble, which
is a posterior sensitivity rather than a controlled sweep because every other parameter is
free to move along the gain axis. Both are reported, and the sweep is the one to quote.

    make gain          # the six runs
    python scripts/23_gain.py

Writes results/gain.json and figures/fig13_gain.png.
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

# The sweep, in the order it is plotted. The published configuration is 0.04.
SWEEP = [
    ("gain_fixed.json", "the gain treated as known"),
    ("gain_tight.json", "half the published uncertainty"),
    ("gain_pub.json", "the published configuration"),
    ("gain_loose.json", "twice the published uncertainty"),
]
FREE = ("gain_free.json", "the gain and the external trend both free")

# The run tag each sweep file was written with, so the posterior ensemble can be reopened.
TAGS = {"gain_fixed.json": "_gfix", "gain_tight.json": "_gtight",
        "gain_pub.json": "_gpub", "gain_loose.json": "_gloose",
        "gain_free.json": "_gfree", "gain_sigma.json": "_gsig"}

# Rows already on disk, used for the identifiability table. The three without a gravity
# leg are the control: if they move the gain, something is wrong with the operator.
IDENT = [("A", "heads only", False), ("D", "evapotranspiration + heads", False),
         ("F", "evapotranspiration + deformation", False),
         ("E", "evapotranspiration + gravity", True),
         ("G", "heads + gravity + deformation", True),
         ("B", "gravity only", True),
         ("H", "all four, coupled closure", True)]


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def identifiability(prior_sd: float = 0.04) -> list[dict]:
    """How much of the gain's prior variance each observing set removes."""
    ia = E.LAYOUT["grace_alpha"]
    out = []
    for key, label, has_gravity in IDENT:
        p = RES / f"posterior_{key}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        X, ok = z["X"], z["ok"] if "ok" in z.files else np.ones(z["X"].shape[1], bool)
        a = np.asarray(X[ia][:, ok]).ravel()
        out.append({"row": key, "label": label, "has_gravity_leg": has_gravity,
                    "alpha_hat": float(a.mean()), "alpha_sd_post": float(a.std()),
                    "alpha_err": float(a.mean() - T.GRACE_ALPHA),
                    "var_removed": float(1.0 - (a.std() / prior_sd) ** 2)})
    return out


def nuisance(tag: str) -> dict:
    """Where the error sits in the degenerate pair, read against withheld truth.

    The gain multiplies the storage change and the external trend adds to it, over a
    record on which storage falls close to linearly. The two are therefore near
    degenerate, and the question a sweep can answer that a single run cannot is which
    of them, and which part of the abstraction estimate, absorbs an error in the
    computed gain.
    """
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
    ap.add_argument("--out", type=str, default="gain.json")
    ap.add_argument("--fig", type=str, default="fig13_gain.png")
    args = ap.parse_args()

    res = {"_operator": "G(t) = alpha * dS(t) / A + external(t) + eps",
           "_prior_shipped": {"mean": 0.85, "sd": 0.04, "box": [0.75, 0.97]},
           "_withheld_truth": {
               "alpha": float(T.GRACE_ALPHA),
               "external_trend_mm_yr": float(T.GRACE_DRIFT[0]),
               "prior_offset": float(0.85 - T.GRACE_ALPHA),
               "prior_offset_in_prior_sd": float((0.85 - T.GRACE_ALPHA) / 0.04),
               "note": ("the twin's gain is not the prior mean. The shipped prior is "
                        "wrong by 1.75 of its own standard deviations on purpose, so "
                        "the sweep measures what a wrongly computed gain costs, not "
                        "what a correct one is worth")}}

    ident = identifiability()
    res["identifiability"] = ident
    no_grav = [r for r in ident if not r["has_gravity_leg"]]
    res["_identifiability_verdict"] = {
        "max_var_removed_without_gravity": max((r["var_removed"] for r in no_grav),
                                               default=float("nan")),
        "var_removed_by_full_closure": next(
            (r["var_removed"] for r in ident if r["row"] == "H"), float("nan")),
        "note": ("the gain is constrained only by the leg it multiplies; "
                 "evapotranspiration, heads and deformation return the prior untouched"),
    }

    rows = []
    for name, label in SWEEP + [FREE]:
        d = load(name)
        if not d or "H" not in d:
            continue
        h, m = d["H"], d["_meta"]
        rows.append({"file": name, "label": label,
                     "alpha_prior_sd": m["alpha_sd"],
                     "drift_trend_sd": m.get("drift_trend_sd"),
                     "free": name == FREE[0],
                     "alpha_hat": h["alpha_hat"], "alpha_sd_post": h["alpha_sd_post"],
                     "var_removed": h["alpha_var_removed"],
                     "mae_mcm": h["mae_mcm"], "mape_pct": h["mape_pct"],
                     "basin_bias_pct": h["basin_bias_pct"],
                     "cover_90": h.get("cover_90"),
                     "crps_mcm": h.get("crps_mcm"),
                     "width90_mcm": h.get("width90_mcm"),
                     "posterior_sensitivity_pct": h.get(
                         "scale_sensitivity_pct_per_prior_sd"),
                     "alpha_Q_corr": h.get("alpha_Q_corr")})
        rows[-1].update(nuisance(TAGS[name]))
    res["sweep"] = rows

    swept = [r for r in rows if not r["free"]]
    free = next((r for r in rows if r["free"]), None)
    if len(swept) >= 2:
        # The controlled propagation: how far the absolute scale moves per unit of gain
        # prior width, read across runs that differ in nothing else.
        x = np.array([r["alpha_prior_sd"] for r in swept])
        b = np.array([r["basin_bias_pct"] for r in swept])
        w = np.array([r["mae_mcm"] for r in swept])
        pub = next((r for r in swept if abs(r["alpha_prior_sd"] - 0.04) < 1e-9), None)
        fix = next((r for r in swept if r["alpha_prior_sd"] <= 0.002), None)
        res["_propagation"] = {
            "bias_span_pts": float(b.max() - b.min()),
            "bias_at_published": pub["basin_bias_pct"] if pub else None,
            "bias_with_gain_known": fix["basin_bias_pct"] if fix else None,
            "cost_of_the_published_assumption_pts": (
                abs(pub["basin_bias_pct"] - fix["basin_bias_pct"])
                if pub and fix else None),
            "mae_span_mcm": float(w.max() - w.min()),
            "slope_bias_pct_per_0p01_sd": float(np.polyfit(x, b, 1)[0] * 0.01),
        }
        if pub and fix:
            res["_propagation"]["width90_with_gain_known_mcm"] = fix.get("width90_mcm")
            res["_propagation"]["width90_at_published_mcm"] = pub.get("width90_mcm")
            if fix.get("width90_mcm"):
                res["_propagation"]["interval_widening_pct"] = float(
                    100.0 * (pub["width90_mcm"] / fix["width90_mcm"] - 1.0))
        if free:
            res["_propagation"]["bias_with_both_free"] = free["basin_bias_pct"]
            res["_propagation"]["free_vs_published_pts"] = float(
                abs(free["basin_bias_pct"] - (pub["basin_bias_pct"] if pub else 0.0)))
            res["_propagation"]["mae_free_over_published"] = float(
                free["mae_mcm"] / pub["mae_mcm"]) if pub else None

        # Where the error goes. The gain, the external trend and the abstraction total
        # all act on the same near-linear signal, so an error in any one of them can be
        # taken up by the other two. The sweep is the only instrument here that can see
        # the split, because it varies one of the three and holds everything else fixed.
        best = min(swept, key=lambda r: r["mae_mcm"])
        res["_partition"] = {
            "alpha_err_with_gain_known": fix.get("alpha_err") if fix else None,
            "alpha_err_at_published": pub.get("alpha_err") if pub else None,
            "alpha_err_with_both_free": free.get("alpha_err") if free else None,
            "trend_err_at_published_mm_yr": pub.get("drift_trend_err") if pub else None,
            "trend_err_with_both_free": free.get("drift_trend_err") if free else None,
            "mae_minimum_at_prior_sd": best["alpha_prior_sd"],
            "mae_minimum_is_the_shipped_configuration": bool(
                abs(best["alpha_prior_sd"] - 0.04) < 1e-9),
            "note": ("a gain prior narrower than the error in the computed gain puts "
                     "that error into the abstraction estimate; releasing the gain "
                     "moves it out of the gain and into the external trend"),
        }

    (RES / args.out).write_text(json.dumps(res, indent=2))

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))

    a0 = ax[0]
    lab = [r["label"].replace(" ", "\n", 1) for r in ident]
    v = [100 * r["var_removed"] for r in ident]
    col = [F.ACCENT if r["has_gravity_leg"] else F.MUTED for r in ident]
    y = np.arange(len(ident))
    a0.barh(y, v, color=col)
    for k, r in enumerate(ident):
        a0.text(max(v[k], 0) + 1.5, y[k], "{:.0f}%".format(v[k]), va="center", fontsize=8)
    a0.set_yticks(y)
    a0.set_yticklabels([r["row"] + "  " + r["label"] for r in ident], fontsize=8)
    a0.invert_yaxis()
    a0.set_xlim(0, max(v) * 1.25)
    a0.set_xlabel("share of the gain's prior variance the posterior removes")
    a0.set_title("Only the leg it multiplies constrains the gain")

    a1 = ax[1]
    if swept:
        x = [r["alpha_prior_sd"] for r in swept]
        b = [r["basin_bias_pct"] for r in swept]
        a1.plot(x, b, marker="o", ms=6, lw=2.0, color=F.ACCENT,
                label="basin abstraction bias, left axis")
        for r in swept:
            a1.annotate("{:+.1f}%".format(r["basin_bias_pct"]),
                        (r["alpha_prior_sd"], r["basin_bias_pct"]),
                        textcoords="offset points", xytext=(9, -12), fontsize=8,
                        color=F.ACCENT)
        pub = next((r for r in swept if abs(r["alpha_prior_sd"] - 0.04) < 1e-9), None)
        if pub:
            a1.axvline(0.04, color=F.WARM, ls=":", lw=1.2)
            a1.annotate("the configuration\nthe entry ships", xy=(0.04, 1.0),
                        xycoords=("data", "axes fraction"),
                        textcoords="offset points", xytext=(6, -26), fontsize=8,
                        color=F.WARM, fontweight="bold")
        if free:
            a1.axhline(free["basin_bias_pct"], color=F.GREEN, ls="-.", lw=1.2)
            a1.annotate("gain and external trend both free: {:+.1f}%".format(
                free["basin_bias_pct"]), (min(x), free["basin_bias_pct"]),
                textcoords="offset points", xytext=(2, 5), fontsize=8,
                color=F.GREEN)
    a1.axhline(0.0, color=F.INK, lw=0.8)
    a1.set_xlabel("width of the gain prior, standard deviations")
    a1.set_ylabel("basin abstraction bias, per cent")
    a1.set_title("What a wrongly computed gain costs, and where the cost goes")
    if swept and all(r.get("width90_mcm") for r in swept):
        a2 = a1.twinx()
        a2.plot([r["alpha_prior_sd"] for r in swept],
                [r["width90_mcm"] for r in swept], marker="s", ms=5, lw=1.6,
                ls="--", color=F.MUTED, label="district 90% interval width, right axis")
        a2.set_ylabel("district 90% interval width, Mm3/yr", color=F.MUTED)
        a2.tick_params(axis="y", colors=F.MUTED)
        a1.set_ylim(min(b) - 1.2,
                    max(max(b), free["basin_bias_pct"] if free else 0) + 2.4)
        h1, l1 = a1.get_legend_handles_labels()
        h2, l2 = a2.get_legend_handles_labels()
        a1.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8, framealpha=0.95)
    else:
        a1.legend(loc="best", fontsize=8)

    F.save(fig, FIG / args.fig)

    # ------------------------------------------------------------------ report
    print("Is the gain identifiable?  Share of its prior variance removed:")
    for r in ident:
        print("  {:3s} {:38s} {:5.0f}%   {}".format(
            r["row"], r["label"], 100 * r["var_removed"],
            "gravity leg present" if r["has_gravity_leg"] else "no gravity leg"))
    print()
    print("What does its uncertainty cost?")
    print("  {:34s} {:>8s} {:>9s} {:>9s} {:>7s}".format(
        "configuration", "prior sd", "MAE", "basin bias", "cov90"))
    for r in rows:
        print("  {:34s} {:8.3f} {:9.2f} {:+8.1f}% {:7.2f}".format(
            r["label"], r["alpha_prior_sd"], r["mae_mcm"], r["basin_bias_pct"],
            r["cover_90"] if r["cover_90"] is not None else float("nan")))
    pr = res.get("_propagation")
    if pr:
        print()
        if pr.get("cost_of_the_published_assumption_pts") is not None:
            print("  moving from a known gain to the published uncertainty moves the "
                  "basin scale by {:.1f} points".format(
                      pr["cost_of_the_published_assumption_pts"]))
        if pr.get("free_vs_published_pts") is not None:
            print("  releasing the gain and the external trend moves it by {:.1f} points"
                  .format(pr["free_vs_published_pts"]))
    pa = res.get("_partition")
    if pa:
        print()
        print("  where the error goes, against a withheld true gain of {:.2f}:".format(
            T.GRACE_ALPHA))
        for r in rows:
            if "alpha_err" not in r:
                continue
            print("    {:34s} gain err {:+.3f}   external trend err {:+.2f} mm/yr"
                  .format(r["label"], r["alpha_err"], r["drift_trend_err"]))
    print("\nwrote results/{} and figures/{}".format(args.out, args.fig))


if __name__ == "__main__":
    main()
