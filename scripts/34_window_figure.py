"""fig21: the window curve on the metered era alone, one panel, for the deck.

Every non-overlapping pair of averaging windows of each length inside 2009-2024, the years
the six-county truth is meter-coded, scored on the change between the two windows. Read
from results/metered_era{tag}.json, so it needs `make metered-era` first; nothing is
recomputed here. Window lengths with fewer than ten pairs are left off the curve.

    python scripts/34_window_figure.py --tag _v3
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

from mizan import figures as F

RES = ROOT / "results"
FIG = ROOT / "figures"
MIN_PAIRS = 10
INK_SOFT = "#6b7078"

# Account key in the results file -> label, colour, marker; the closure is drawn last.
SERIES = [
    ("FLAT", "mapped area × acre-foot", F.MUTED, "D"),
    ("WATERBAL", "+ precipitation deficit", F.SAND, "s"),
    ("OPENLOOP", "published open loop, ET / 0.80", F.WARM, "^"),
    ("CLOSURE", "the closure", F.ACCENT, "o"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="_v3")
    args = ap.parse_args()

    res = json.loads((RES / f"metered_era{args.tag}.json").read_text())
    era = res["_meta"]["era_year0"]
    last = res["_meta"]["years"][1]
    curve = res["window_curve_metered_era"]["by_window_years"]
    ws = sorted(int(w) for w, v in curve.items() if v["n_pairs"] >= MIN_PAIRS)
    err = {k: np.array([curve[str(w)]["mean_abs_error_pts"][k] for w in ws])
           for k, *_ in SERIES}

    fig, ax = plt.subplots(figsize=(8.6, 3.5))
    for k, lab, col, mk in SERIES:
        closure = k == "CLOSURE"
        ax.plot(ws, err[k], marker=mk, ms=6 if closure else 5, lw=2.4 if closure else 1.4,
                color=col, label=lab, zorder=3 if closure else 2)

    # First window length at which the closure is closer than every meter-free account.
    rivals = np.vstack([err[k] for k, *_ in SERIES if k != "CLOSURE"])
    ahead = np.nonzero(err["CLOSURE"] < rivals.min(axis=0))[0]
    if ahead.size:
        xc = ws[ahead[0]]
        ax.axvline(xc, color=INK_SOFT, lw=1.0, ls=":")
        ax.annotate(f"from {xc} years the closure\nleads every meter-free account",
                    (xc, err["CLOSURE"][ahead[0]]), textcoords="offset points",
                    xytext=(12, -64), fontsize=9, color=F.ACCENT, fontweight="bold")

    ax.set_xticks(ws)
    ax.set_xticklabels([f"{w}\n{curve[str(w)]['n_pairs']} pairs" for w in ws])
    ax.set_xlabel("length of each averaging window, years")
    ax.set_ylabel("mean error on the change, points")
    ax.set_title(f"Six Kansas counties, metered {era}–{last}: error on the change "
                 "between two periods")
    ax.set_ylim(0, None)
    ax.legend(loc="lower left", fontsize=8.5, ncol=2)
    F.despine(ax)

    out = FIG / ("fig21_window_metered.png" if args.tag == "_v3"
                 else f"fig21_window_metered{args.tag}.png")
    F.save(fig, out)
    print("wrote " + str(out.relative_to(ROOT)))
    for i, w in enumerate(ws):
        print("  {} yr ({:>2} pairs)  ".format(w, curve[str(w)]["n_pairs"])
              + "  ".join("{}={:.2f}".format(k, err[k][i]) for k, *_ in SERIES))


if __name__ == "__main__":
    main()
