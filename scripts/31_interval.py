"""The one prediction the blind transfer failed, measured and corrected out of sample.

The transfer's pre-registered prediction P2 asked the 90 per cent interval to cover 0.80
to 0.97 of new county-years. It covered 0.98: **the posterior is too wide away from the
block its error budget was estimated on**, and the excess is there at the 50 and 80 per
cent levels too. An account that over-covers overstates its own uncertainty, which is the
safe direction and still a miscalibration.

This script does three things and nothing else.

1. **Measures it.** Per block, the single multiplicative factor on the ensemble spread
   about its own mean that makes the 90 per cent interval cover exactly 0.90. A factor
   below one is an interval that was too wide.
2. **Tests whether one factor exists.** If the four blocks want four different factors the
   defect is a property of each block and no rule fixes it. If they want nearly the same
   one, it is a property of the error budget and a rule does.
3. **Scores the rule out of sample, leave-one-block-out.** The factor applied to a block
   is fitted on the other three only, so the block being scored never enters its own
   calibration. That is the only form in which a calibration rule can be reported here,
   because the alternative is the in-sample factor, which is an oracle.

The rule is a reporting instrument and it is deliberately not fed back into the
inversion: the posteriors stay exactly as the blind runs wrote them, and the shipped
scores are the uncalibrated ones. What this measures is what a calibration would buy on
the next basin.

    python scripts/31_interval.py [--tag _v3]

Writes results/interval{tag}.json and figures/fig18_interval.png.
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

LEVELS = (0.50, 0.80, 0.90)


def rescale(ens: np.ndarray, f: float) -> np.ndarray:
    """The ensemble spread about its own mean, multiplied by `f`.

    The mean is untouched, so the point estimate and every level score are unchanged and
    only the interval moves. Abstraction is positive, so the rescaling is applied in the
    log where the inversion parameterises it.
    """
    L = np.log(np.maximum(ens, 1.0))
    m = L.mean(axis=0, keepdims=True)
    return np.exp(m + f * (L - m))


def coverage_at(ens: np.ndarray, q: np.ndarray, f: float, lv: float = 0.90) -> float:
    e = rescale(ens, f)
    lo = np.quantile(e, 0.5 - lv / 2.0, axis=0)
    hi = np.quantile(e, 0.5 + lv / 2.0, axis=0)
    return float(((q >= lo) & (q <= hi)).mean())


def fit_factor(ens: np.ndarray, q: np.ndarray, lv: float = 0.90,
               grid=np.linspace(0.05, 2.0, 196)) -> float:
    """The spread factor whose `lv` interval covers `lv` of the county-years.

    Coverage is a step function of the factor, so this takes the smallest factor whose
    coverage reaches the nominal level, which is the shortest interval that is not
    under-covering.
    """
    cov = np.array([coverage_at(ens, q, f, lv) for f in grid])
    ok = np.nonzero(cov >= lv)[0]
    return float(grid[ok[0]]) if ok.size else float(grid[-1])


def scores(ens: np.ndarray, q: np.ndarray, f: float) -> dict:
    e = rescale(ens, f)
    return {"factor": f, **MT.coverage(e, q), "crps_mcm": MT.crps(e, q),
            "mape_pct": float((np.abs(e.mean(axis=0) - q) / q).mean() * 100.0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="_v3")
    args = ap.parse_args()

    tr = json.loads((RES / f"transfer{args.tag}.json").read_text())
    drv = importlib.import_module("11_kansas_run")
    sweepmod = importlib.import_module("27_metered_era")

    blocks = ["gmd4a"] + list(tr["pooled_new_counties"]["blocks"])
    data = {}
    for blk in blocks:
        R.set_block(blk)
        years = np.arange(K.YEAR0, K.YEAR1 + 1)
        era = years >= tr["blocks"][blk]["metered_era_year0"]
        q = K.reported_annual()[0].sum(axis=0)[:, era]
        sfx = "" if blk == "gmd4a" else "_" + blk
        ens = np.load(RES / f"kansas_posterior_ETH{args.tag}{sfx}.npz")["ens"][..., era]
        data[blk] = (ens, q)
    R.set_block("gmd4a")

    out = {"_meta": {"tag": args.tag, "blocks": blocks,
                     "what": ("the multiplicative factor on the posterior spread that "
                              "makes the 90 per cent interval nominal; below one means "
                              "the interval was too wide"),
                     "rule": ("leave-one-block-out: the factor applied to a block is the "
                              "mean of the factors fitted on the other blocks, so no "
                              "block enters its own calibration")}}

    # ---------------------------------------------------------------- 1. measure it
    out["per_block"] = {}
    for blk, (ens, q) in data.items():
        f = fit_factor(ens, q)
        row = {"label": tr["blocks"][blk]["label"], "n_county_years": int(q.size),
               "uncalibrated": scores(ens, q, 1.0),
               "in_sample_factor": f,
               "in_sample": scores(ens, q, f)}
        row["cover_by_level_uncalibrated"] = {
            str(int(100 * lv)): coverage_at(ens, q, 1.0, lv) for lv in LEVELS}
        out["per_block"][blk] = row

    f_all = np.array([out["per_block"][b]["in_sample_factor"] for b in blocks])
    out["factor_spread"] = {
        "by_block": {b: round(float(x), 3) for b, x in zip(blocks, f_all)},
        "mean": float(f_all.mean()), "sd": float(f_all.std(ddof=1)),
        "min": float(f_all.min()), "max": float(f_all.max()),
        "reading": ("one factor exists if the spread is small against the distance from "
                    "one; the numbers are reported either way")}

    # ------------------------------------------------- 3. score the rule out of sample
    out["leave_one_block_out"] = {}
    for blk, (ens, q) in data.items():
        others = [b for b in blocks if b != blk]
        f = float(np.mean([out["per_block"][b]["in_sample_factor"] for b in others]))
        out["leave_one_block_out"][blk] = {
            "factor_from": others, "factor": f, **scores(ens, q, f)}

    # The same rule fitted only on the blocks the method had never seen, which is the
    # cleaner population: the published block is where the error budget was estimated in
    # sample, so its own factor of one is not evidence about a new basin.
    new = list(tr["pooled_new_counties"]["blocks"])
    out["leave_one_new_block_out"] = {}
    for blk in new:
        others = [b for b in new if b != blk]
        f = float(np.mean([out["per_block"][b]["in_sample_factor"] for b in others]))
        out["leave_one_new_block_out"][blk] = {
            "factor_from": others, "factor": f, **scores(*data[blk], f)}

    def pooled(key):
        n = sum(out["per_block"][b]["n_county_years"] for b in new)
        return {
            k: float(sum(out[key][b][k] * out["per_block"][b]["n_county_years"]
                         for b in new) / n)
            for k in ("cover_50", "cover_80", "cover_90", "crps_mcm")}
    out["pooled_new_counties"] = {
        "leave_one_new_block_out": pooled("leave_one_new_block_out"),
        "n_county_years": sum(out["per_block"][b]["n_county_years"] for b in new),
        "uncalibrated": {k: float(sum(
            out["per_block"][b]["uncalibrated"][k] * out["per_block"][b]["n_county_years"]
            for b in new) / sum(out["per_block"][b]["n_county_years"] for b in new))
            for k in ("cover_50", "cover_80", "cover_90", "crps_mcm", "width90_mcm")},
        "leave_one_block_out": pooled("leave_one_block_out"),
    }

    # The honest counterfactual: a rule fitted on the published block alone, which is all
    # the entry had before the transfer. It is the rule a reader would have had to trust.
    f_home = out["per_block"]["gmd4a"]["in_sample_factor"]
    out["home_block_rule"] = {
        "factor": f_home,
        "by_block": {b: scores(*data[b], f_home) for b in new}}

    (RES / f"interval{args.tag}.json").write_text(json.dumps(out, indent=2))

    # ------------------------------------------------------------------- the figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
        grid = np.linspace(0.2, 1.3, 45)
        colors = {"gmd4a": "#9fb7c9", "west": "#2b6ca3", "gmd3w": "#d9a441",
                  "gmd3e": "#b04a3c"}
        for blk, (ens, q) in data.items():
            ax[0].plot(grid, [coverage_at(ens, q, f) for f in grid],
                       color=colors[blk], lw=2,
                       ls="--" if blk == "gmd4a" else "-",
                       label=tr["blocks"][blk]["label"])
        ax[0].axhline(0.90, color="k", lw=1, ls=":")
        ax[0].set_xlabel("multiplier on the posterior spread")
        ax[0].set_ylabel("share of county-years inside the 90% interval")
        ax[0].set_title("Every block wants a narrower interval than it has", fontsize=10)
        ax[0].legend(fontsize=7.5, loc="lower right")

        keys = ["uncalibrated", "leave_one_block_out"]
        x = np.arange(len(new))
        for i, k in enumerate(keys):
            vals = [out[k][b]["cover_90"] if k != "uncalibrated"
                    else out["per_block"][b]["uncalibrated"]["cover_90"] for b in new]
            ax[1].bar(x + (i - 0.5) * 0.38, vals, 0.38,
                      color=("#9fb7c9", "#2b6ca3")[i],
                      label=("as run", "leave-one-block-out calibration")[i])
        ax[1].axhline(0.90, color="k", lw=1, ls=":")
        ax[1].set_xticks(x)
        short = {"west": "West Kansas (GMD4 south, GMD1)", "gmd3w": "GMD3 west",
                 "gmd3e": "GMD3 east"}
        ax[1].set_xticklabels([short.get(b, b) for b in new], fontsize=9)
        ax[1].set_ylim(0.6, 1.02)
        ax[1].set_ylabel("90% interval coverage")
        ax[1].set_title("The correction scored on blocks it was not fitted on",
                        fontsize=10)
        ax[1].legend(fontsize=8, loc="lower left")
        pn = out["pooled_new_counties"]
        ax[1].text(0.5, 0.995, "pooled over {} county-years: 0.98 to {:.2f} at 90 per "
                               "cent, {:.2f} to {:.2f} at 50".format(
                       pn["n_county_years"], pn["leave_one_block_out"]["cover_90"],
                       pn["uncalibrated"]["cover_50"],
                       pn["leave_one_block_out"]["cover_50"]),
                   transform=ax[1].transAxes, ha="center", va="top", fontsize=8.5,
                   color="#1b4a70")
        fig.tight_layout()
        FIG.mkdir(exist_ok=True)
        fig.savefig(FIG / "fig18_interval.png", dpi=160)
    except ImportError:
        pass

    # ------------------------------------------------------------------- the report
    print("Spread factor that makes the 90 per cent interval nominal "
          "(below 1 = the interval was too wide):\n")
    print("  {:<34s} {:>7s} {:>9s} {:>9s} {:>9s} {:>9s}".format(
        "block", "factor", "cover50", "cover80", "cover90", "CRPS"))
    for blk in blocks:
        r = out["per_block"][blk]
        u = r["uncalibrated"]
        print("  {:<34s} {:7.2f} {:9.2f} {:9.2f} {:9.2f} {:9.2f}".format(
            r["label"][:34], r["in_sample_factor"], u["cover_50"], u["cover_80"],
            u["cover_90"], u["crps_mcm"]))
    fs = out["factor_spread"]
    print("\n  factors {} -> mean {:.2f}, sd {:.2f}".format(
        fs["by_block"], fs["mean"], fs["sd"]))

    print("\nLeave-one-block-out: the factor applied to a block comes from the others\n")
    print("  {:<34s} {:>7s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s}".format(
        "block", "factor", "cover50", "cover80", "cover90", "CRPS", "was"))
    for blk in blocks:
        r = out["leave_one_block_out"][blk]
        u = out["per_block"][blk]["uncalibrated"]
        print("  {:<34s} {:7.2f} {:9.2f} {:9.2f} {:9.2f} {:9.2f} {:9.2f}".format(
            out["per_block"][blk]["label"][:34], r["factor"], r["cover_50"],
            r["cover_80"], r["cover_90"], r["crps_mcm"], u["crps_mcm"]))

    p = out["pooled_new_counties"]
    print("\nPooled over the {} county-years of the blocks never seen:".format(
        p["n_county_years"]))
    for k, lab in (("uncalibrated", "as run"),
                   ("leave_one_block_out", "calibrated"),
                   ("leave_one_new_block_out", "new blocks only")):
        v = p[k]
        print("  {:<12s} cover 50/80/90 {:.2f} {:.2f} {:.2f}   CRPS {:.2f}".format(
            lab, v["cover_50"], v["cover_80"], v["cover_90"], v["crps_mcm"]))

    print("\nThe rule a reader would have had before the transfer, fitted on the "
          "published block alone (factor {:.2f}):".format(out["home_block_rule"]["factor"]))
    for b, v in out["home_block_rule"]["by_block"].items():
        print("  {:<34s} cover90 {:.2f}  CRPS {:.2f}".format(
            out["per_block"][b]["label"][:34], v["cover_90"], v["crps_mcm"]))

    print("\nwrote results/interval{}.json".format(args.tag))


if __name__ == "__main__":
    main()
