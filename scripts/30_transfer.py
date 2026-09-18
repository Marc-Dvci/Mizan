"""L2 Kansas: score the frozen closure on county blocks it never saw.

The published block is six counties, so every error bar clustered by county rests on
n = 6. This script scores the same configuration, frozen, on three further blocks of six
counties each, chosen and fetched before their meters were opened, with the predictions
recorded in `DECISION_LOG.md` and committed before any use file was downloaded
(`10_kansas_fetch.py` refuses the use files until that commit exists).

Every account is scored on each block's own metered era, found from the WIMAS
measurement codes by the rule of `27_metered_era.py`, against the per-point truth. The
transfer result is then pooled over the eighteen new counties, clustered by county, and
reported beside the published block rather than mixed with it.

    python scripts/30_transfer.py [--tag _v3] [--arms _v5]

Writes results/transfer{tag}.json and figures/fig17_transfer.png.
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

BLOCK_ORDER = ("gmd4a",) + K.TRANSFER_BLOCKS
W = 5
BARS = ("FLAT", "WATERBAL", "OPENLOOP")
LABEL = {"CLOSURE": "closure, evapotranspiration + heads",
         "ET": "evapotranspiration leg alone",
         "FLAT": "mapped irrigated area x one acre-foot per acre",
         "WATERBAL": "the same, plus half the year's precipitation deficit",
         "OPENLOOP": "unmixed evapotranspiration over a fixed efficiency of 0.80"}


def metered_era_year(q_by: np.ndarray, years: np.ndarray, floor: float = 0.98,
                     county_floor: float = 0.95):
    """First year from which every later year is meter-coded, block and county.

    The rule the published block's era was held to (`27_metered_era.py` and its guard):
    above 98 per cent of the block's reported volume in every later year, and above 95
    in every county. A county-year with no reported volume does not count against it.
    """
    tot = q_by.sum(axis=0)
    blk = q_by[0].sum(axis=0) / np.where(tot.sum(axis=0) > 0, tot.sum(axis=0), 1.0)
    cty = np.where(tot > 0, q_by[0] / np.where(tot > 0, tot, 1.0), 1.0)
    for i, y in enumerate(years):
        if (blk[i:] > floor).all() and (cty[:, i:] > county_floor).all():
            return int(y)
    return None


def county_gain(acc: dict, q: np.ndarray, against: str) -> np.ndarray:
    """Points of relative error the closure removes from one bar, per county."""
    rel = lambda v: (np.abs(v - q) / q).mean(axis=1) * 100.0
    return rel(acc[against]) - rel(acc["CLOSURE"])


def clustered(gain: np.ndarray) -> dict:
    n = gain.size
    se = float(gain.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"points": float(gain.mean()), "se_by_county": se,
            "n_se_by_county": float(gain.mean() / se) if se > 0 else float("nan"),
            "n_counties": int(n), "n_counties_favouring_closure": int((gain > 0).sum())}


def score_block(block: str, tag: str, arms: list, sweep) -> dict:
    drv = importlib.import_module("11_kansas_run")
    a = drv.assemble(pool=True, block=block)
    years = np.arange(K.YEAR0, K.YEAR1 + 1)
    q_by, share, meta = K.reported_annual()
    q_all = q_by.sum(axis=0)
    era0 = metered_era_year(q_by, years)
    era = years >= (era0 if era0 is not None else K.YEAR1 + 1)
    if era.sum() < 2 * W + 1:
        # A block whose codes never settle is scored on the years its block-wide share
        # clears the floor, and the report says so.
        blk_share = q_by[0].sum(axis=0) / q_all.sum(axis=0)
        era0 = int(next(y for y, s in zip(years, blk_share) if s > 0.98))
        era = years >= era0
        era_rule = "block share only"
    else:
        era_rule = "block and every county"
    q = q_all[:, era]

    P = K.precipitation()
    deficit = P.mean(axis=1, keepdims=True) - P
    pts_all = {"FLAT": a["irr_area"] * R.PRIOR_DEPTH_M,
               "WATERBAL": a["irr_area"] * (R.PRIOR_DEPTH_M + 0.5 * deficit / 1000.0),
               "OPENLOOP": a["et_obs"] / 0.80}
    pts = {k: v[:, era] for k, v in pts_all.items()}

    post = {}
    for arm in [tag] + arms:
        for row in ("ETH", "ET"):
            f = RES / f"kansas_posterior_{row}{arm}{'' if block == 'gmd4a' else '_' + block}.npz"
            if f.exists():
                post[(arm, row)] = np.load(f)["ens"]
    if (tag, "ETH") not in post:
        raise FileNotFoundError(f"no {tag} ETH posterior for block {block}")

    out = {"label": K.BLOCKS[block]["label"], "counties": list(K.COUNTIES),
           "names": [K.COUNTY_NAME[c] for c in K.COUNTIES],
           "grid": [a["region"].nrow, a["region"].ncol],
           "active_cells": int((a["region"].county >= 0).sum()),
           "irrigated_km2": float(a["irr_area"].mean(axis=1).sum() / 1e6),
           "n_wells": len(a["wl"]["wells"]),
           "metered_era_year0": era0, "metered_era_rule": era_rule,
           "n_county_years": int(q.size),
           "metered_share_by_year": {int(y): float(q_by[0, :, i].sum() / q_all[:, i].sum())
                                     for i, y in enumerate(years)},
           "metered_mcm_yr_era": float(q.sum(axis=0).mean() / 1e6),
           "metered_depth_m_era": float(
               (q.sum(axis=0) / a["irr_area"][:, era].sum(axis=0)).mean()),
           "arms": {}}

    for arm in [tag] + arms:
        if (arm, "ETH") not in post:
            continue
        ens = post[(arm, "ETH")][..., era]
        acc = {"CLOSURE": ens.mean(axis=0), **pts}
        if (arm, "ET") in post:
            acc["ET"] = post[(arm, "ET")][..., era].mean(axis=0)
        # Two oracles no practitioner has: each rule with its constant fitted to this
        # block's own meters. They measure how much of a rule's error is its constant.
        grid = np.linspace(0.05, 1.0, 951)
        d_star = float(grid[int(np.argmin([np.abs(a["irr_area"][:, era] * g - q).mean()
                                           for g in grid]))])
        acc["FLAT_ORACLE"] = a["irr_area"][:, era] * d_star
        grid = np.linspace(0.2, 1.6, 1401)
        e_star = float(grid[int(np.argmin([np.abs(a["et_obs"][:, era] / g - q).mean()
                                           for g in grid]))])
        acc["OPENLOOP_ORACLE"] = a["et_obs"][:, era] / e_star
        LABEL["FLAT_ORACLE"] = (f"area x depth, depth fitted to this block's meters "
                                f"({d_star / 0.3048:.2f} af/acre)")
        LABEL["OPENLOOP_ORACLE"] = (f"open loop, efficiency fitted to this block's "
                                    f"meters ({e_star:.2f})")
        level = {k: {"label": LABEL[k], **MT.point_scores(v, q),
                     "rel_err_by_county": {c: round(float(x), 1) for c, x in zip(
                         K.COUNTIES, (np.abs(v - q) / q).mean(axis=1) * 100.0)}}
                 for k, v in acc.items()}
        level["CLOSURE"].update(MT.coverage(ens, q))
        level["CLOSURE"]["crps_mcm"] = MT.crps(ens, q)
        if (arm, "ET") in post:
            level["ET"].update(MT.coverage(post[(arm, "ET")][..., era], q))
        gains = {}
        for b in BARS:
            g = county_gain(acc, q, b)
            gains[b] = {**clustered(g),
                        "gain_by_county": {c: round(float(x), 1)
                                           for c, x in zip(K.COUNTIES, g)},
                        "closure_closer_pct_of_county_years": float(
                            100.0 * (np.abs(acc["CLOSURE"] - q) < np.abs(acc[b] - q)).mean())}
        s5, _, _, _ = sweep(ens, pts, q, years[era], W)
        best_bar = min(s5["mean_abs_error_pts"][b] for b in BARS)
        out["arms"][arm] = {
            "level": level, "level_gain_vs": gains,
            "change_5yr": {**s5,
                           "margin_over_best_bar_pts": best_bar
                           - s5["mean_abs_error_pts"]["CLOSURE"]},
            "best_level_account": min(level, key=lambda k: level[k]["mape_pct"]),
        }
    return out


def pooled(res: dict, blocks, tag: str) -> dict:
    """Gains pooled over the counties of `blocks`, clustered by county."""
    out = {"blocks": list(blocks), "n_counties": 0}
    for b in BARS:
        g = np.concatenate([
            np.array(list(res[blk]["arms"][tag]["level_gain_vs"][b]["gain_by_county"]
                          .values())) for blk in blocks])
        out[b] = clustered(g)
        out["n_counties"] = int(g.size)
    lv = {}
    for k in ("CLOSURE", "ET", "FLAT", "WATERBAL", "OPENLOOP"):
        vals = [res[blk]["arms"][tag]["level"][k]["mape_pct"] for blk in blocks
                if k in res[blk]["arms"][tag]["level"]]
        if vals:
            lv[k] = float(np.mean(vals))
    out["mean_block_mape_pct"] = lv
    out["cover_90_mean"] = float(np.mean(
        [res[blk]["arms"][tag]["level"]["CLOSURE"]["cover_90"] for blk in blocks]))
    out["change_5yr_mean_abs_error_pts"] = {
        k: float(np.mean([res[blk]["arms"][tag]["change_5yr"]["mean_abs_error_pts"][k]
                          for blk in blocks]))
        for k in ("CLOSURE",) + BARS}
    return out


def predictions(res: dict, tag: str, arms: list) -> dict:
    """The pre-registered predictions, each scored as written in the decision log."""
    new = list(K.TRANSFER_BLOCKS)
    have = [b for b in new if b in res]
    P = {}
    # P1: closure beats the open loop on the level in at least two thirds of new counties
    g = np.concatenate([np.array(list(
        res[b]["arms"][tag]["level_gain_vs"]["OPENLOOP"]["gain_by_county"].values()))
        for b in have])
    P["P1"] = {"statement": "closure has lower relative error than the open loop on the "
                            "level in at least two thirds of the new counties",
               "value": f"{int((g > 0).sum())} of {g.size}",
               "pass": bool((g > 0).mean() >= 2.0 / 3.0)}
    # P2: 90% coverage in 0.80 to 0.97 of new county-years
    cov = np.concatenate([np.full(res[b]["n_county_years"],
                                  res[b]["arms"][tag]["level"]["CLOSURE"]["cover_90"])
                          for b in have])
    P["P2"] = {"statement": "the 90 per cent interval covers the metered value in 0.80 "
                            "to 0.97 of new county-years",
               "value": round(float(cov.mean()), 3),
               "pass": bool(0.80 <= cov.mean() <= 0.97)}
    # P3: on the five-year change, closure beats the open loop, pooled over new blocks
    c = np.mean([res[b]["arms"][tag]["change_5yr"]["mean_abs_error_pts"]["CLOSURE"]
                 for b in have])
    o = np.mean([res[b]["arms"][tag]["change_5yr"]["mean_abs_error_pts"]["OPENLOOP"]
                 for b in have])
    P["P3"] = {"statement": "on the five-year change the closure beats the open loop, "
                            "mean over the new blocks",
               "value": f"closure {c:.2f} vs open loop {o:.2f} points",
               "pass": bool(c < o)}
    # P4: _v5 beats _v3 on the change, pooled over new blocks
    if arms and all(arms[0] in res[b]["arms"] for b in have):
        c5 = np.mean([res[b]["arms"][arms[0]]["change_5yr"]["mean_abs_error_pts"]["CLOSURE"]
                      for b in have])
        P["P4"] = {"statement": f"{arms[0]} beats {tag} on the five-year change, mean "
                                "over the new blocks",
                   "value": f"{arms[0]} {c5:.2f} vs {tag} {c:.2f} points",
                   "pass": bool(c5 < c)}
    # P5: area x one acre-foot degrades in GMD3 relative to the published block
    if "gmd4a" in res and any(b.startswith("gmd3") for b in have):
        f4 = res["gmd4a"]["arms"][tag]["level"]["FLAT"]["mape_pct"]
        f3 = np.mean([res[b]["arms"][tag]["level"]["FLAT"]["mape_pct"]
                      for b in have if b.startswith("gmd3")])
        d3 = np.mean([res[b]["metered_depth_m_era"] for b in have if b.startswith("gmd3")])
        P["P5"] = {"statement": "area x one acre-foot per acre has a larger relative "
                                "error in GMD3 than on the published block, because "
                                "the applied depth there is not the northwest's",
                   "value": f"GMD3 {f3:.1f}% vs GMD4 north {f4:.1f}%; metered depth "
                            f"in GMD3 {d3 / 0.3048:.2f} acre-feet per acre",
                   "pass": bool(f3 > f4)}
    return P


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="_v3")
    ap.add_argument("--arms", type=str, default="_v5",
                    help="secondary configurations to report beside the primary")
    ap.add_argument("--blocks", type=str, default=",".join(BLOCK_ORDER))
    args = ap.parse_args()
    arms = [x for x in args.arms.split(",") if x]
    sweep = importlib.import_module("27_metered_era").sweep

    res = {}
    for block in args.blocks.split(","):
        try:
            res[block] = score_block(block, args.tag, arms, sweep)
        except FileNotFoundError as exc:
            print(f"  {block}: skipped ({exc})")
    R.set_block("gmd4a")

    new = [b for b in K.TRANSFER_BLOCKS if b in res]
    out = {"_meta": {"tag": args.tag, "arms": arms, "window_years": W,
                     "error_bars": "clustered by county, the unit a transfer claim "
                                   "generalises over"},
           "blocks": res,
           "pooled_new_counties": pooled(res, new, args.tag) if new else None,
           "pooled_all_counties": pooled(res, list(res), args.tag),
           "predictions": predictions(res, args.tag, arms) if new else {}}
    for arm in arms:
        if new and all(arm in res[b]["arms"] for b in new):
            out[f"pooled_new_counties{arm}"] = pooled(res, new, arm)
    (RES / f"transfer{args.tag}.json").write_text(json.dumps(out, indent=2))

    # ------------------------------------------------------------------ the figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(14, 4.6),
                               gridspec_kw={"width_ratios": [3, 2]})
        colors = {"gmd4a": "#9fb7c9", "west": "#2b6ca3", "gmd3w": "#d9a441",
                  "gmd3e": "#b04a3c"}
        x = 0
        ticks, labels = [], []
        for blk in [b for b in BLOCK_ORDER if b in res]:
            g = res[blk]["arms"][args.tag]["level_gain_vs"]["OPENLOOP"]["gain_by_county"]
            for name, v in zip(res[blk]["names"], g.values()):
                ax[0].bar(x, v, color=colors[blk],
                          hatch="//" if blk == "gmd4a" else None, edgecolor="white")
                ticks.append(x)
                labels.append(name)
                x += 1
            x += 1
        ax[0].axhline(0, color="k", lw=0.8)
        ax[0].set_xticks(ticks)
        ax[0].set_xticklabels(labels, rotation=90, fontsize=7.5)
        ax[0].set_ylabel("points of relative error removed\nfrom the open loop, per county")
        ax[0].set_title("Hatched: the block the method was built on. "
                        "Solid: blocks it never saw", fontsize=10)
        from matplotlib.patches import Patch
        ax[0].legend([Patch(color=colors[b]) for b in BLOCK_ORDER if b in res],
                     [res[b]["label"] for b in BLOCK_ORDER if b in res], fontsize=7.5)

        keys = ["CLOSURE", "ET", "FLAT", "WATERBAL", "OPENLOOP"]
        xx = np.arange(len(keys))
        wdt = 0.8 / max(len(res), 1)
        for i, blk in enumerate([b for b in BLOCK_ORDER if b in res]):
            lv = res[blk]["arms"][args.tag]["level"]
            ax[1].bar(xx + (i - len(res) / 2 + 0.5) * wdt,
                      [lv[k]["mape_pct"] if k in lv else np.nan for k in keys], wdt,
                      color=colors[blk], hatch="//" if blk == "gmd4a" else None,
                      edgecolor="white")
        ax[1].set_xticks(xx)
        ax[1].set_xticklabels(["closure", "ET leg", "area x depth", "+ weather",
                               "open loop"], fontsize=8.5)
        ax[1].set_ylabel("relative error on the level, %")
        ax[1].set_title("Every account, every block, its own metered era", fontsize=10)
        fig.tight_layout()
        FIG.mkdir(exist_ok=True)
        fig.savefig(FIG / "fig17_transfer.png", dpi=160)
    except ImportError:
        pass

    # ------------------------------------------------------------------ the report
    for blk in [b for b in BLOCK_ORDER if b in res]:
        r = res[blk]
        print(f"\n{blk}: {r['label']}")
        print(f"  grid {r['grid'][0]}x{r['grid'][1]}, {r['active_cells']} active cells, "
              f"{r['irrigated_km2']:,.0f} km2 irrigated, {r['n_wells']} wells; metered "
              f"era from {r['metered_era_year0']} ({r['metered_era_rule']}), "
              f"{r['n_county_years']} county-years, {r['metered_mcm_yr_era']:,.0f} "
              f"Mm3/yr, {r['metered_depth_m_era'] / 0.3048:.2f} af/acre")
        for arm, A in r["arms"].items():
            print(f"  {arm}: level MAPE  " + "  ".join(
                f"{k} {v['mape_pct']:.1f}%" for k, v in A["level"].items())
                + f"  cover90 {A['level']['CLOSURE']['cover_90']:.2f}")
            print(f"  {arm}: five-year change |err|  " + "  ".join(
                f"{k} {v:.1f}" for k, v in A["change_5yr"]["mean_abs_error_pts"].items())
                + f"  margin {A['change_5yr']['margin_over_best_bar_pts']:+.2f}")
            for b in BARS:
                g = A["level_gain_vs"][b]
                print(f"  {arm}: gain vs {b:8s} {g['points']:+6.1f} +- {g['se_by_county']:4.1f} "
                      f"({g['n_se_by_county']:+.1f} se) {g['n_counties_favouring_closure']}/"
                      f"{g['n_counties']} counties  {g['gain_by_county']}")

    for key in ("pooled_new_counties", "pooled_all_counties"):
        p = out[key]
        if not p:
            continue
        print(f"\n{key} ({p['n_counties']} counties, {args.tag}):")
        for b in BARS:
            g = p[b]
            print(f"  gain vs {b:8s} {g['points']:+6.1f} +- {g['se_by_county']:4.1f} by county "
                  f"({g['n_se_by_county']:+.1f} se), {g['n_counties_favouring_closure']}/"
                  f"{g['n_counties']} counties favour the closure")
        print("  mean block MAPE: " + "  ".join(
            f"{k} {v:.1f}%" for k, v in p["mean_block_mape_pct"].items()))
        print(f"  closure 90% coverage {p['cover_90_mean']:.2f}; five-year change |err| "
              + "  ".join(f"{k} {v:.1f}" for k, v in p["change_5yr_mean_abs_error_pts"].items()))

    print("\npre-registered predictions:")
    for k, v in out["predictions"].items():
        print(f"  {k} {'PASS' if v['pass'] else 'FAIL'}  {v['statement']}: {v['value']}")
    print(f"\nwrote results/transfer{args.tag}.json")


if __name__ == "__main__":
    main()
