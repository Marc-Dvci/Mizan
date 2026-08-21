"""fig12: the change is the question, and the arithmetic that wins on level cannot answer it.

Three panels, left to right the argument.

* **Left.** Every one of the 136 non-overlapping five-year window pairs the record admits.
  Each point is one pair: the water-balance bar's absolute error on the change against the
  closure's. Below the diagonal the closure is closer. The three contrasts the Kansas
  policy record names are marked.
* **Centre.** The same accounts on the two metrics that matter, level and change, so the
  reversal between them is visible in one image.
* **Right.** Where the interannual variance of each account comes from. The metered record
  is 56 per cent weather; the water-balance bar is 92 per cent weather and keeps almost
  none of the trend that is left when weather is removed.

    python scripts/22_verify_figure.py --tag _v4
"""
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mizan import figures as F, ks_data as K, ks_run as R

RES = ROOT / "results"
FIG = ROOT / "figures"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="_v4")
    ap.add_argument("--out", type=str, default="fig12_verify.png")
    ap.add_argument("--layout", choices=("banner", "grid"), default="banner",
                    help="banner is one row of four, for a page-width figure in the "
                         "documents; grid is two by two, which fills a 16:9 slide and "
                         "doubles the size of every panel")
    args = ap.parse_args()

    drv = importlib.import_module("11_kansas_run")
    a = drv.assemble(pool=True)
    q, _ = K.metered_annual()
    q = q / 1e6
    years = np.arange(K.YEAR0, K.YEAR1 + 1)
    area = a["irr_area"]
    P = K.precipitation()
    dfc = P.mean(axis=1, keepdims=True) - P
    flat = area * R.PRIOR_DEPTH_M / 1e6
    wb = area * (R.PRIOR_DEPTH_M + 0.5 * dfc / 1000.0) / 1e6
    ol = a["et_obs"] / 0.80 / 1e6
    ens = np.load(RES / ("kansas_posterior_ETH" + args.tag + ".npz"))["ens"] / 1e6
    est = ens.mean(axis=0)

    def C(v, A, B):
        return 100.0 * (v[..., B].mean(-1).sum(-1) / v[..., A].mean(-1).sum(-1) - 1.0)

    W = 5
    starts = range(0, len(years) - W + 1)
    pairs = [(i, j) for i, j in itertools.combinations(starts, 2) if j >= i + W]
    named = {(5, 13): "SD-6", (13, 18): "GMD4", (0, 20): "long"}
    ec, ew, marks = [], [], []
    for i, j in pairs:
        A = np.zeros(len(years), bool); A[i:i + W] = True
        B = np.zeros(len(years), bool); B[j:j + W] = True
        t = C(q, A, B)
        ec.append(abs(C(ens, A, B).mean() - t))
        ew.append(abs(C(wb, A, B) - t))
        if (i, j) in named:
            marks.append((ec[-1], ew[-1], named[(i, j)]))
    ec, ew = np.array(ec), np.array(ew)

    if args.layout == "grid":
        fig, axg = plt.subplots(2, 2, figsize=(12.6, 8.0))
        ax = axg.ravel()
        fig.subplots_adjust(hspace=0.34, wspace=0.26)
    else:
        fig, ax = plt.subplots(1, 4, figsize=(16.4, 4.0))

    # ---------------------------------------------------------------- left
    a0 = ax[0]
    below = ec < ew
    a0.scatter(ec[below], ew[below], s=16, c=F.ACCENT, alpha=0.75, lw=0,
               label="closure closer ({} of {})".format(int(below.sum()), len(pairs)))
    a0.scatter(ec[~below], ew[~below], s=16, c=F.MUTED, alpha=0.75, lw=0,
               label="water balance closer ({})".format(int((~below).sum())))
    m = max(ec.max(), ew.max()) * 1.05
    a0.plot([0, m], [0, m], color=F.INK, lw=0.9, ls="--", zorder=1)
    for x, y, lab in marks:
        a0.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 5),
                    fontsize=8, color=F.WARM, fontweight="bold")
        a0.scatter([x], [y], s=42, facecolor="none", edgecolor=F.WARM, lw=1.4, zorder=3)
    a0.set_xlim(0, m); a0.set_ylim(0, m)
    a0.set_xlabel("closure error on the change, points")
    a0.set_ylabel("water-balance error, points")
    a0.set_title("Every five-year window pair in the record")
    a0.legend(loc="lower right", fontsize=8)

    # ---------------------------------------------------------------- centre
    a1 = ax[1]
    rows = [("mapped area\nx acre-foot", flat, F.MUTED),
            ("+ half the\nprecip deficit", wb, F.SAND),
            ("evapotranspiration\n/ 0.80", ol, F.GREEN),
            ("the closure", est, F.ACCENT)]
    lv = [np.abs(v - q).mean() for _, v, _ in rows]
    ch = []
    for _, v, _ in rows:
        e = []
        for i, j in pairs:
            A = np.zeros(len(years), bool); A[i:i + W] = True
            B = np.zeros(len(years), bool); B[j:j + W] = True
            e.append(abs(C(v, A, B) - C(q, A, B)))
        ch.append(np.mean(e))
    y = np.arange(len(rows))
    a1.barh(y - 0.19, lv, height=0.36, color=[c for _, _, c in rows], alpha=0.45,
            label="level, Mm3/yr")
    a1.barh(y + 0.19, ch, height=0.36, color=[c for _, _, c in rows],
            label="change, points")
    for k in range(len(rows)):
        a1.text(lv[k] + 0.3, y[k] - 0.19, "{:.1f}".format(lv[k]), va="center", fontsize=8)
        a1.text(ch[k] + 0.3, y[k] + 0.19, "{:.1f}".format(ch[k]), va="center",
                fontsize=8, fontweight="bold")
    a1.set_yticks(y); a1.set_yticklabels([r[0] for r in rows], fontsize=8)
    a1.invert_yaxis()
    a1.set_xlim(0, max(max(lv), max(ch)) * 1.32)
    a1.set_xlabel("error against the withheld meters")
    a1.set_title("The two metrics rank them differently")
    a1.legend(loc="lower right", fontsize=8)

    # ---------------------------------------------------------------- right
    a2 = ax[2]
    Pb = P.mean(axis=0); Pb = (Pb - Pb.mean()) / Pb.std()
    t = (years - years.mean()) / years.std()
    names, wshare, trend = [], [], []
    for nm, v in [("metered", q), ("closure", est), ("water\nbalance", wb),
                  ("mapped\narea", flat)]:
        b = v.sum(0) - v.sum(0).mean()
        fit = np.polyval(np.polyfit(Pb, b, 1), Pb)
        names.append(nm)
        wshare.append(100 * (1 - ((b - fit) ** 2).sum() / (b ** 2).sum()))
        trend.append(abs(np.polyfit(t, b - fit, 1)[0]))
    x = np.arange(len(names))
    a2.bar(x - 0.19, wshare, width=0.36, color=F.SAND, label="weather share, %")
    a2.bar(x + 0.19, trend, width=0.36, color=F.ACCENT,
           label="trend left after weather,\nMm3 per sd-year")
    for k in range(len(names)):
        a2.text(x[k] - 0.19, wshare[k] + 1.5, "{:.0f}".format(wshare[k]),
                ha="center", fontsize=8)
        a2.text(x[k] + 0.19, trend[k] + 1.5, "{:.0f}".format(trend[k]),
                ha="center", fontsize=8, fontweight="bold")
    a2.set_xticks(x); a2.set_xticklabels(names, fontsize=8)
    a2.set_ylim(0, max(max(wshare), max(trend)) * 1.42)
    a2.set_title("What each account is made of")
    a2.legend(loc="upper left", fontsize=7.5)

    # ---------------------------------------------------------------- far right
    a3 = ax[3]
    ws = list(range(2, 9))
    series = {"the closure": ([], F.ACCENT, "o"),
              "+ half the precip deficit": ([], F.SAND, "s"),
              "evapotranspiration / 0.80": ([], F.GREEN, "^"),
              "mapped area x acre-foot": ([], F.MUTED, "d")}
    src = {"the closure": est, "+ half the precip deficit": wb,
           "evapotranspiration / 0.80": ol, "mapped area x acre-foot": flat}
    for w in ws:
        st = range(0, len(years) - w + 1)
        pp = [(i, j) for i, j in itertools.combinations(st, 2) if j >= i + w]
        for nm, (acc, _, _) in series.items():
            e = []
            for i, j in pp:
                A = np.zeros(len(years), bool); A[i:i + w] = True
                B = np.zeros(len(years), bool); B[j:j + w] = True
                t = C(q, A, B)
                v = C(ens, A, B).mean() if nm == "the closure" else C(src[nm], A, B)
                e.append(abs(v - t))
            acc.append(np.mean(e))
    for nm, (acc, col, mk) in series.items():
        a3.plot(ws, acc, marker=mk, ms=5, lw=2.0 if nm == "the closure" else 1.3,
                color=col, label=nm)
    # Where the closure passes the strongest meter-free bar.
    bar = np.array(series["+ half the precip deficit"][0])
    clo = np.array(series["the closure"][0])
    hit = np.nonzero(clo < bar)[0]
    if hit.size:
        xc = ws[hit[0]]
        a3.axvline(xc, color=F.WARM, lw=1.0, ls=":")
        a3.annotate("closing the loop\nstarts paying at\n{} years".format(xc),
                    (xc, max(clo.max(), bar.max()) * 0.86), fontsize=8,
                    color=F.WARM, fontweight="bold",
                    textcoords="offset points", xytext=(7, 0))
    a3.set_xlabel("length of the averaging window, years")
    a3.set_ylabel("error on the change, points")
    a3.set_title("The aquifer integrates; weather does not")
    a3.set_ylim(0, None)
    a3.legend(loc="lower left", fontsize=7.5)

    F.save(fig, FIG / args.out)
    print("wrote figures/" + args.out)
    print("  closure closer on {} of {} pairs".format(int(below.sum()), len(pairs)))
    print("  level  " + "  ".join("{}={:.1f}".format(r[0].replace(chr(10), " "), v)
                                  for r, v in zip(rows, lv)))
    print("  change " + "  ".join("{}={:.1f}".format(r[0].replace(chr(10), " "), v)
                                  for r, v in zip(rows, ch)))


if __name__ == "__main__":
    main()
