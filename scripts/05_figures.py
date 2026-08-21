"""Every figure and table in the submission, regenerated from the frozen results.

Usage:  python scripts/05_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt

from mizan import config as C, forcing as F, fields, metrics as MT
from mizan import figures as FG

RES = ROOT / "results"
FIG = ROOT / "figures"
ORDER = ["ET", "A", "B", "C", "D", "E", "F", "G", "SAT", "HS3", "HS", "H",
         "M", "HM1", "HM3"]
SHORT = {"ET": "ET only", "A": "heads", "B": "gravity", "C": "deformation",
         "D": "ET+heads", "E": "ET+gravity", "F": "ET+deform.",
         "G": "heads+grav.+def.", "H": "all four",
         "M": "meters only", "HM1": "all four + 1 meter", "HM3": "all four + 3 meters",
         "SAT": "satellites, no wells", "HS": "all four, 10 wells",
         "HS3": "all four, 3 wells"}


def fig_basin(tr):
    """The synthetic basin: irrigated area, subsidence, drawdown."""
    mask = F.pivot_mask(C.TRUTH)
    fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.4))
    ext = [0, C.DOMAIN_KM, 0, C.DOMAIN_KM]

    ax[0].imshow(mask, origin="lower", extent=ext, cmap="Greens", vmin=0, vmax=1.6)
    for k in (1, 2):
        ax[0].axhline(k * C.DOMAIN_KM / 3, color=FG.INK, lw=0.7)
        ax[0].axvline(k * C.DOMAIN_KM / 3, color=FG.INK, lw=0.7)
    for d in range(C.NDIST):
        i, j = divmod(d, 3)
        ax[0].text((j + 0.5) * C.DOMAIN_KM / 3, (i + 0.5) * C.DOMAIN_KM / 3, f"D{d}",
                   ha="center", va="center", color=FG.INK, fontsize=8, alpha=0.55)
    ax[0].set_title(f"irrigated land, {int(mask.sum())} km$^2$ in 9 districts")

    im = ax[1].imshow(tr["subsidence_final"] * 100, origin="lower", extent=ext,
                      cmap="magma_r")
    plt.colorbar(im, ax=ax[1], label="cm")
    ax[1].set_title(f"subsidence after {C.NYEAR} yr, peak "
                    f"{tr['subsidence_final'].max()*100:.0f} cm")

    im = ax[2].imshow(C.H_INIT - tr["head_final"], origin="lower", extent=ext,
                      cmap="Blues")
    plt.colorbar(im, ax=ax[2], label="m")
    ax[2].set_title(f"drawdown, peak {(C.H_INIT - tr['head_final']).max():.0f} m")
    for a in ax:
        a.set_xlabel("km")
        a.grid(False)
    ax[0].set_ylabel("km")
    return FG.save(fig, FIG / "fig1_basin.png")


def fig_ablation(ab):
    """The invention slide's evidence: what each leg is worth."""
    rows = [k for k in ORDER if k in ab]
    mae = [ab[k]["mae_mcm"] for k in rows]
    base = ab["BASELINE"]["mae_mcm"]
    oracle = ab["BASELINE_ORACLE"]["mae_mcm"]
    col = [FG.ACCENT if k in ("H", "SAT") else FG.GREEN if k.startswith("HM") or k == "M"
           else FG.SAND if k.startswith("HS") else FG.MUTED for k in rows]

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    b = ax.bar(range(len(rows)), mae, color=col, width=0.68)
    ax.axhline(base, color=FG.WARM, lw=1.6,
               label=f"published open loop, fixed 0.80 ({base:.1f})")
    ax.axhline(oracle, color=FG.WARM, lw=1.2, ls="--",
               label=f"open loop, efficiency fitted to the answer ({oracle:.1f})")
    swatch = [
        (FG.MUTED, "one or two legs"),
        (FG.ACCENT, "the closure, and the closure without wells"),
        (FG.SAND, "the closure on a sparse well network"),
        (FG.GREEN, "metering, alone or added to the closure"),
    ]
    handles = [plt.matplotlib.patches.Patch(facecolor=c, label=t) for c, t in swatch]
    for r, v in zip(b, mae):
        ax.text(r.get_x() + r.get_width() / 2, v * 1.04, f"{v:.1f}", ha="center",
                va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(4, 1500)
    ax.set_yticks([5, 10, 20, 50, 100, 200])
    ax.get_yaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([SHORT[k] for k in rows], rotation=32, ha="right")
    ax.set_ylabel("district-annual abstraction MAE, Mm$^3$/yr (log)")
    ax.set_title("What each observation is worth, against withheld truth")
    lines = ax.get_legend_handles_labels()[0]
    ax.legend(handles=lines + handles, loc="upper left", fontsize=7.6, ncol=2,
              columnspacing=1.4, handlelength=1.8)
    FG.despine(ax)
    return FG.save(fig, FIG / "fig2_ablation.png")


def fig_calibration(ab):
    """Coverage: a ninety per cent interval has to contain the answer nine times in ten."""
    rows = [k for k in ORDER if k in ab and "cover_90" in ab[k]]
    lv = [50, 80, 90]
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for k in rows:
        y = [ab[k][f"cover_{l}"] * 100 for l in lv]
        ax[0].plot(lv, y, marker="o", ms=4,
                   color=FG.ACCENT if k == "H" else FG.MUTED,
                   lw=2.0 if k == "H" else 0.9, label=SHORT[k] if k == "H" else None)
    ax[0].plot([40, 95], [40, 95], color=FG.INK, ls=":", lw=1.0, label="nominal")
    ax[0].set_xlabel("nominal interval, %")
    ax[0].set_ylabel("empirical coverage, %")
    ax[0].set_title("Interval calibration")
    ax[0].legend(fontsize=8)
    FG.despine(ax[0])

    w = [ab[k]["width90_mcm"] for k in rows]
    m = [ab[k]["mae_mcm"] for k in rows]
    ax[1].scatter(w, m, color=[FG.ACCENT if k == "H" else FG.MUTED for k in rows], s=34)
    # Log axes, because the single-leg rows sit an order of magnitude away from the
    # closure rows and a linear frame collapses the cluster that matters into one blob.
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    # Label the rows the argument turns on. The remaining rows sit inside the same
    # cluster and are left unlabelled rather than stacked on top of each other.
    off = {"H": (7, -3), "SAT": (7, 2), "ET": (-6, 7), "D": (7, -9),
           "G": (7, 2), "M": (7, -3), "A": (7, 2), "B": (-10, 8), "C": (-46, 5)}
    for k, x, y in zip(rows, w, m):
        if k not in off:
            continue
        ax[1].annotate(SHORT[k], (x, y), fontsize=7.0,
                       color=FG.ACCENT if k == "H" else FG.MUTED,
                       xytext=off[k], textcoords="offset points")
    ax[1].set_ylim(min(m) * 0.72, max(m) * 1.9)
    ax[1].set_xlim(min(w) * 0.80, max(w) * 1.5)
    ax[1].set_xticks([60, 100, 200, 400])
    ax[1].set_yticks([5, 10, 20, 50, 100, 200])
    for a in (ax[1].get_xaxis(), ax[1].get_yaxis()):
        a.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        a.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    ax[1].set_xlabel("mean 90% interval width, Mm$^3$/yr")
    ax[1].set_ylabel("MAE, Mm$^3$/yr")
    ax[1].set_title("Sharpness against accuracy")
    FG.despine(ax[1])
    return FG.save(fig, FIG / "fig3_calibration.png")


def fig_scatter(tr):
    """Estimate against truth, district-annual, with the open-loop account beside it."""
    post = np.load(RES / "posterior_H.npz")
    ens = post["ens"]
    q = tr["q_annual"] / 1e6
    hat = ens.mean(axis=0) / 1e6
    lo = np.quantile(ens, 0.05, axis=0) / 1e6
    hi = np.quantile(ens, 0.95, axis=0) / 1e6
    ol = MT.baseline_open_loop(tr["obs_et"]) / 1e6

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 4.0), sharex=True, sharey=True)
    lim = [0, max(q.max(), hat.max(), ol.max()) * 1.06]
    for a, y, e, t in ((ax[0], ol, None, "open loop, efficiency fixed at 0.80"),
                       (ax[1], hat, (hat - lo, hi - hat), "Mizan closure, 90% interval")):
        a.plot(lim, lim, color=FG.INK, ls=":", lw=1.0)
        if e is None:
            a.scatter(q, y, s=16, color=FG.WARM, alpha=0.75)
        else:
            a.errorbar(q.ravel(), y.ravel(), yerr=np.vstack([e[0].ravel(), e[1].ravel()]),
                       fmt="o", ms=3.2, lw=0.6, color=FG.ACCENT, alpha=0.8)
        a.set_xlim(lim); a.set_ylim(lim)
        a.set_xlabel("true district abstraction, Mm$^3$/yr")
        a.set_title(t)
        FG.despine(a)
    ax[0].set_ylabel("estimated, Mm$^3$/yr")
    return FG.save(fig, FIG / "fig4_scatter.png")


def fig_closure(tr):
    """The accounting identity, and the residual the open-loop account leaves behind."""
    q = tr["q_annual"].sum(axis=0) / 1e9
    et = MT.et_annual(tr["obs_et"]).sum(axis=0) / 1e9
    ol = MT.baseline_open_loop(tr["obs_et"]).sum(axis=0) / 1e9
    post = np.load(RES / "posterior_H.npz")["ens"].sum(axis=1) / 1e9
    yr = np.arange(C.START_YEAR, C.START_YEAR + C.NYEAR)

    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax[0].fill_between(yr, np.quantile(post, 0.05, axis=0), np.quantile(post, 0.95, axis=0),
                       color=FG.ACCENT, alpha=0.20, label="Mizan, 90%")
    ax[0].plot(yr, post.mean(axis=0), color=FG.ACCENT, lw=2.0, label="Mizan")
    ax[0].plot(yr, q, color=FG.INK, lw=1.6, ls="--", label="truth")
    ax[0].plot(yr, ol, color=FG.WARM, lw=1.4, label="open loop")
    ax[0].plot(yr, et, color=FG.MUTED, lw=1.2, label="satellite ET, as retrieved")
    ax[0].set_ylabel("basin abstraction, km$^3$/yr")
    ax[0].set_title("The basin account")
    ax[0].legend(fontsize=8)
    FG.despine(ax[0])

    ax[1].bar(yr - 0.2, (ol - q), width=0.4, color=FG.WARM, label="open loop")
    ax[1].bar(yr + 0.2, (post.mean(axis=0) - q), width=0.4, color=FG.ACCENT, label="Mizan")
    ax[1].axhline(0, color=FG.INK, lw=1.0)
    ax[1].set_ylabel("closure residual, km$^3$/yr")
    ax[1].set_title("What the account fails to explain")
    ax[1].legend(fontsize=8)
    FG.despine(ax[1])
    return FG.save(fig, FIG / "fig5_closure.png")


def fig_nullspace():
    """What the observation set could not resolve, stated rather than hidden."""
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for k in ORDER:
        p = RES / f"posterior_{k}.npz"
        if not p.exists():
            continue
        r = np.load(p)["variance_ratio"]
        ax.plot(np.arange(1, r.size + 1), np.clip(r, 0, 1.2),
                color=FG.ACCENT if k == "H" else FG.MUTED,
                lw=2.0 if k == "H" else 0.8, label=SHORT[k] if k in ("H", "ET") else None)
    ax.axhline(0.10, color=FG.WARM, ls="--", lw=1.0)
    ax.text(2, 0.12, "resolved to 90%", color=FG.WARM, fontsize=8)
    ax.set_xlabel("direction of the 180-dimensional district-year abstraction vector")
    ax.set_ylabel("share of prior variance remaining")
    ax.set_title("What the satellites can and cannot see")
    ax.legend(fontsize=8)
    FG.despine(ax)
    return FG.save(fig, FIG / "fig6_nullspace.png")


def fig_allocation():
    """What the posterior lets a regulator decide: how much can be taken, and whether
    where it is taken matters."""
    al = json.loads((RES / "allocation.json").read_text())
    d = np.load(RES / "allocation.npz")
    fr = al["frontier"]
    x = np.array([f["delivered_km3"] for f in fr])
    mu = np.array([f["mean_mcm"] for f in fr])
    lo = np.array([f["p10_mcm"] for f in fr])
    hi = np.array([f["p90_mcm"] for f in fr])

    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.8))

    ax[0].fill_between(x, lo, hi, color=FG.ACCENT, alpha=0.20,
                       label="posterior 10-90%")
    ax[0].plot(x, mu, color=FG.ACCENT, lw=2.2, marker="o", ms=4, label="mean")
    ax[0].axhline(al["hist_permanent_loss_mcm"], color=FG.WARM, ls="--", lw=1.2)
    ax[0].text(x.min(), al["hist_permanent_loss_mcm"], " already lost over the record",
               color=FG.WARM, fontsize=7.5, va="bottom")
    ax[0].set_xlabel("water delivered over 20 years, km$^3$")
    ax[0].set_ylabel("storage capacity destroyed, Mm$^3$")
    ax[0].set_title("What the next twenty years cost")
    ax[0].legend(fontsize=8)
    FG.despine(ax[0])

    names = [k for k in ("uniform", "risk_bounded", "chance_constrained") if k in al]
    lbl = {"uniform": "uniform cut", "risk_bounded": "risk-bounded",
           "chance_constrained": "chance-constrained"}
    xs = np.arange(len(names))
    ax[1].bar(xs - 0.19, [al[k]["loss_cvar90_mcm"] for k in names], width=0.36,
              color=FG.MUTED, label="zone surrogate")
    ax[1].bar(xs + 0.19, [al[k]["simulator"]["cvar90_mcm"] for k in names], width=0.36,
              color=FG.ACCENT, label="full MODFLOW")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([lbl[k] for k in names], rotation=16, ha="right")
    ax[1].set_ylabel("permanent loss, Mm$^3$ (90% tail)")
    ax[1].set_title(f"At equal water, {al['uniform']['delivered_km3']:.1f} km$^3$")
    ax[1].legend(fontsize=8)
    FG.despine(ax[1])

    du = d["q_uniform"].sum(axis=1) / 1e6
    ax[2].bar(np.arange(C.NDIST) - 0.19, du, width=0.38, color=FG.WARM, label="uniform")
    if d["q_opt"].size:
        ax[2].bar(np.arange(C.NDIST) + 0.19, d["q_opt"].sum(axis=1) / 1e6, width=0.38,
                  color=FG.ACCENT, label="risk-bounded")
    ax[2].set_xticks(range(C.NDIST))
    ax[2].set_xticklabels([f"D{i}" for i in range(C.NDIST)])
    ax[2].set_ylabel("20-year quota, Mm$^3$")
    ax[2].set_title(f"Where it is taken ({al['reallocation_gain_pct']:+.1f}% on loss)")
    ax[2].legend(fontsize=8)
    FG.despine(ax[2])
    fig.tight_layout()
    return FG.save(fig, FIG / "fig7_allocation.png")


def fig_voi():
    """The ranking depends on the forecast, and that is the finding."""
    v = json.loads((RES / "voi.json").read_text())
    d = np.load(RES / "voi.npz")
    cc = d["cand_cells"]
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.0))

    lab = {"fc_q_last5": "how much is being pumped",
           "fc_perm_loss": "how much capacity is being destroyed"}
    colr = {"fc_q_last5": FG.ACCENT, "fc_perm_loss": FG.SAND}
    for key in ("fc_q_last5", "fc_perm_loss"):
        if key not in v:
            continue
        cur = np.array(v[key]["greedy_curve"])
        n_meter = sum(1 for k in v[key]["greedy_order"] if k.startswith("meter"))
        ax[0].plot(np.arange(cur.size), cur / cur[0] * 100, marker="o", ms=3.5,
                   color=colr[key], lw=2.0,
                   label=f"{lab[key]}: {(1-cur[-1]/cur[0])*100:.0f}% removed, "
                         f"{n_meter} of {cur.size-1} are meters")
    ax[0].set_xlabel("instruments installed, in the order data worth ranks them")
    ax[0].set_ylabel("remaining forecast uncertainty, % of today")
    ax[0].set_title("What each new instrument buys")
    ax[0].legend(fontsize=7.5, loc="center right")
    FG.despine(ax[0])

    mask = fields.upscale_mask(F.pivot_mask(C.TRUTH), 2)
    ax[1].imshow(mask, origin="lower", extent=[0, C.DOMAIN_KM, 0, C.DOMAIN_KM],
                 cmap="Greens", vmin=0, vmax=2.6)
    third = C.DOMAIN_KM / 3.0
    for k in (1, 2):
        ax[1].axhline(k * third, color=FG.INK, lw=0.6, alpha=0.5)
        ax[1].axvline(k * third, color=FG.INK, lw=0.6, alpha=0.5)

    # Geodetic sites can rank close together in space. Nudge any marker that would sit
    # on top of one already drawn, so the rank stays readable.
    geo = [k for k in v["fc_perm_loss"]["greedy_order"] if not k.startswith("meter")]
    placed = []
    for r, n in enumerate(geo[:8]):
        kind, j = n.rsplit("_", 1)
        rr, cl = cc[int(j)]
        x = (cl + 0.5) * C.EST.delr_m / 1000.0
        y = (rr + 0.5) * C.EST.delr_m / 1000.0
        for _ in range(24):
            if all((x - px) ** 2 + (y - py) ** 2 > 7.0 ** 2 for px, py in placed):
                break
            x, y = x + 2.4, y + 2.4
        placed.append((x, y))
        ax[1].scatter([x], [y], s=170, marker="o" if kind == "piezo" else "^",
                      facecolor="white", edgecolor=FG.ACCENT, lw=1.5, zorder=3)
        ax[1].text(x, y, str(r + 1), fontsize=7.5, ha="center", va="center", zorder=4)

    # The district's meter rank goes in whichever corner is farthest from a station.
    order_m = [k for k in v["fc_q_last5"]["greedy_order"] if k.startswith("meter")]
    rank = {int(k.split("_d")[1]): i + 1 for i, k in enumerate(order_m)}
    for dd, r in rank.items():
        i, j = divmod(dd, 3)
        corners = [(j * third + 1.6, (i + 1) * third - 1.8, "left", "top"),
                   ((j + 1) * third - 1.6, (i + 1) * third - 1.8, "right", "top"),
                   (j * third + 1.6, i * third + 1.8, "left", "bottom"),
                   ((j + 1) * third - 1.6, i * third + 1.8, "right", "bottom")]
        cx, cy, ha, va = max(
            corners,
            key=lambda c: min([(c[0] - px) ** 2 + (c[1] - py) ** 2
                               for px, py in placed] or [1e9]))
        ax[1].text(cx, cy, f"meter {r}", ha=ha, va=va, fontsize=8.5,
                   color=FG.WARM, weight="bold")

    ax[1].scatter([], [], s=110, marker="o", facecolor="white", edgecolor=FG.ACCENT,
                  lw=1.5, label="piezometer")
    ax[1].scatter([], [], s=110, marker="^", facecolor="white", edgecolor=FG.ACCENT,
                  lw=1.5, label="geodetic station")
    ax[1].legend(loc="lower right", fontsize=7.5, framealpha=0.9, frameon=True)
    ax[1].set_xlim(0, C.DOMAIN_KM)
    ax[1].set_ylim(0, C.DOMAIN_KM)
    ax[1].set_title("Where the next instrument goes: meter rank for\n"
                    "abstraction, geodesy for capacity")
    ax[1].set_xlabel("km")
    ax[1].set_ylabel("km")
    ax[1].grid(False)
    fig.tight_layout()
    return FG.save(fig, FIG / "fig8_voi.png")


def fig_identity():
    """The accounting identity, assembling, with the residual named.

    Slide 5 of the deck. Every other figure reports a number; this one states the
    mechanism, because the mechanism is what is being claimed.
    """
    import matplotlib.patches as mp

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    legs = [
        ("satellite\nevapotranspiration", "what left the surface", FG.GREEN),
        ("basin\ngravimetry", "what left the basin", FG.ACCENT),
        ("interferometric\ndisplacement", "how the skeleton reacted", FG.SAND),
        ("heads at wells,\nwhere they exist", "local drawdown in time", FG.MUTED),
    ]
    for i, (name, sub, col) in enumerate(legs):
        y = 4.58 - i * 0.94
        ax.add_patch(mp.FancyBboxPatch((0.15, y - 0.36), 2.5, 0.74,
                                       boxstyle="round,pad=0.04,rounding_size=0.08",
                                       facecolor="white", edgecolor=col, lw=1.6))
        ax.text(1.40, y + 0.08, name, ha="center", va="center", fontsize=8.6,
                color=FG.INK, weight="bold")
        ax.text(1.40, y - 0.24, sub, ha="center", va="center", fontsize=7.2,
                color=FG.MUTED)
        ax.annotate("", xy=(4.05, 2.5), xytext=(2.72, y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5,
                                    connectionstyle="arc3,rad=0.06"))

    ax.add_patch(mp.FancyBboxPatch((4.10, 1.30), 2.55, 2.40,
                                   boxstyle="round,pad=0.05,rounding_size=0.10",
                                   facecolor="#f4f6f8", edgecolor=FG.INK, lw=1.8))
    ax.text(5.38, 3.42, "one aquifer", ha="center", fontsize=8.6, weight="bold")
    ax.text(5.38, 2.86, "MODFLOW 6 + CSUB", ha="center", fontsize=8.2, color=FG.ACCENT)
    ax.text(5.38, 2.44, "mass balance", ha="center", fontsize=8.0)
    ax.text(5.38, 2.10, "stress-strain law", ha="center", fontsize=8.0)
    ax.text(5.38, 1.62, "every leg must agree\nwith every other", ha="center",
            fontsize=7.4, color=FG.MUTED)

    ax.annotate("", xy=(7.35, 2.5), xytext=(6.70, 2.5),
                arrowprops=dict(arrowstyle="-|>", color=FG.INK, lw=1.8))
    ax.add_patch(mp.FancyBboxPatch((7.40, 1.70), 2.45, 1.60,
                                   boxstyle="round,pad=0.05,rounding_size=0.10",
                                   facecolor="white", edgecolor=FG.WARM, lw=2.2))
    ax.text(8.62, 2.92, "the one term nobody", ha="center", fontsize=8.0)
    ax.text(8.62, 2.62, "measures", ha="center", fontsize=8.0)
    ax.text(8.62, 2.16, "abstraction", ha="center", fontsize=11.5, weight="bold",
            color=FG.WARM)
    ax.text(8.62, 1.88, "per district, per year, with an interval", ha="center",
            fontsize=6.8, color=FG.MUTED)

    ax.text(5.0, 0.78,
            "storage change from gravimetry and compaction, consumptive use from "
            "evapotranspiration,\nand drawdown from heads are three different functions "
            "of the same abstraction field.",
            ha="center", va="center", fontsize=8.2, color=FG.INK)
    ax.text(5.0, 0.22,
            "Forcing them to agree leaves the abstraction field and the consumptive "
            "fraction identified together.",
            ha="center", va="center", fontsize=8.2, color=FG.WARM, weight="bold")
    return FG.save(fig, FIG / "fig5b_identity.png")


def fig_context():
    """The national account the entry opens with, in the Kingdom's own published figures."""
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.5))

    # GASTAT, Water Accounts Publication 2023, released 5 January 2025.
    years = [2022, 2023]
    nonren = [10849 / 0.94, 10849.0]
    agri = [9356 / 0.93, 9356.0]
    x = np.arange(len(years))
    ax[0].bar(x - 0.18, nonren, width=0.34, color=FG.MUTED,
              label="non-renewable groundwater extracted")
    ax[0].bar(x + 0.18, agri, width=0.34, color=FG.ACCENT,
              label="of which agricultural")
    for xi, v in zip(x - 0.18, nonren):
        ax[0].text(xi, v * 1.02, f"{v:,.0f}", ha="center", fontsize=7.5)
    for xi, v in zip(x + 0.18, agri):
        ax[0].text(xi, v * 1.02, f"{v:,.0f}", ha="center", fontsize=7.5)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([str(y) for y in years])
    ax[0].set_ylabel("Mm$^3$/yr")
    ax[0].set_ylim(0, 14000)
    ax[0].set_title("Saudi non-renewable groundwater, down 6% and 7%")
    ax[0].legend(fontsize=7.5, loc="upper right")
    FG.despine(ax[0])
    ax[0].set_xlabel("GASTAT, Water Accounts Publication 2023. The 2022 bars are\n"
                     "the reported 2023 figures raised by the reported decreases.",
                     fontsize=6.9, color=FG.MUTED)

    # MEWA, National Water Strategy: irrigation efficiency today against best practice,
    # and the constant the published open-loop method assumes instead.
    names = ["MEWA:\nirrigation efficiency\ntoday",
             "published method:\nassumed\nefficiency",
             "MEWA:\nbest practice"]
    vals = [50, 80, 75]
    cols = [FG.WARM, FG.MUTED, FG.GREEN]
    ax[1].bar(range(3), vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax[1].text(i, v + 1.5, f"{v}%", ha="center", fontsize=9, weight="bold")
    ax[1].set_xticks(range(3))
    ax[1].set_xticklabels(names, fontsize=7.4)
    ax[1].set_ylim(0, 95)
    ax[1].set_ylabel("per cent")
    ax[1].set_title("The constant nobody measures")
    FG.despine(ax[1])
    ax[1].set_xlabel("MEWA National Water Strategy; López Valencia et al. 2020. The\n"
                     "two definitions are not identical, which is the point: the number\n"
                     "an abstraction estimate divides by is assumed, never measured.",
                     fontsize=6.9, color=FG.MUTED)

    fig.tight_layout()
    return FG.save(fig, FIG / "fig0_context.png")


def fig_detection():
    """A district whose books do not balance, and how far clear of the rest it stands."""
    v = json.loads((RES / "detection.json").read_text())
    ratio = np.load(RES / "detection.npz")["ratio"]
    d = int(v["hidden_district"])
    other = np.delete(ratio, d)

    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.6))
    col = [FG.WARM if i == d else FG.MUTED for i in range(ratio.size)]
    ax[0].bar(range(ratio.size), ratio, color=col, width=0.66)
    ax[0].axhspan(other.min(), other.max(), color=FG.MUTED, alpha=0.16,
                  label=f"the other eight, {other.min():.2f} to {other.max():.2f}")
    ax[0].text(d, ratio[d] + 0.05, f"{ratio[d]:.2f}", ha="center", fontsize=9,
               weight="bold", color=FG.WARM)
    ax[0].set_xticks(range(ratio.size))
    ax[0].set_xticklabels([f"D{i}" for i in range(ratio.size)])
    ax[0].set_ylabel("closure estimate / what consumptive use explains")
    ax[0].set_title(f"District {d} is {v['z_score']:.1f} standard deviations clear")
    ax[0].set_ylim(0, ratio.max() * 1.22)
    ax[0].legend(fontsize=7.5, loc="upper right")
    FG.despine(ax[0])

    names = ["open-loop account", "Mizan closure"]
    vals = [v["open_loop_recovery_pct"], v["closure_recovery_pct"]]
    ax[1].bar(range(2), vals, color=[FG.MUTED, FG.ACCENT], width=0.5)
    for i, val in enumerate(vals):
        ax[1].text(i, val + 2, f"{val:.0f}%", ha="center", fontsize=11, weight="bold")
    ax[1].set_xticks(range(2))
    ax[1].set_xticklabels(names)
    ax[1].set_ylim(0, 108)
    ax[1].set_ylabel("share of that district's true abstraction attributed")
    ax[1].set_title(f"{v['hidden_mcm_per_year']:.0f} Mm$^3$/yr planted with no canopy, "
                    f"{100*v['hidden_share_of_district']:.0f}% of the district")
    FG.despine(ax[1])
    fig.tight_layout()
    return FG.save(fig, FIG / "fig9_detection.png")


def fig_kansas():
    """L2: the same closure on real observations, against per-well metered pumping."""
    from mizan import ks_data as K

    v = json.loads((RES / "kansas_v3.json").read_text())
    d = np.load(RES / "kansas_posterior_ETH_v3.npz")
    q_true = d["q_true"] / 1e6
    ens = d["ens"] / 1e6
    hat = ens.mean(axis=0)
    lo = np.quantile(ens, 0.05, axis=0)
    hi = np.quantile(ens, 0.95, axis=0)
    ol = d["et_obs"] / 0.80 / 1e6
    years = np.arange(K.YEAR0, K.YEAR1 + 1)

    fig = plt.figure(figsize=(10.6, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.42, wspace=0.30)

    # the block, and where the irrigation is
    ax = fig.add_subplot(gs[0, 0])
    frac = d["frac"].mean(axis=0)
    county = d["county"]
    ax.imshow(np.where(county >= 0, frac, np.nan), origin="lower", cmap="Greens",
              vmin=0, vmax=0.5)
    ax.contour(county, levels=np.arange(-0.5, 6, 1), colors=FG.INK, linewidths=0.6)
    for i, c in enumerate(K.COUNTIES):
        rr, cc = np.nonzero(county == i)
        ax.text(cc.mean(), rr.mean(), K.COUNTY_NAME[c], ha="center", va="center",
                fontsize=7.0, color=FG.INK)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(f"Northwest Kansas, {v['_meta']['irrigated_km2']:,.0f} km$^2$ irrigated")

    # the estimate against the meters
    ax = fig.add_subplot(gs[0, 1:])
    lim = [0, max(q_true.max(), hat.max(), ol.max()) * 1.05]
    ax.plot(lim, lim, color=FG.INK, ls=":", lw=1.0)
    ax.scatter(q_true.ravel(), ol.ravel(), s=13, color=FG.WARM, alpha=0.55,
               label=f"open loop at 0.80 ({v['BASELINE']['mae_mcm']:.1f})")
    ax.errorbar(q_true.ravel(), hat.ravel(),
                yerr=np.vstack([(hat - lo).ravel(), (hi - hat).ravel()]),
                fmt="o", ms=3.0, lw=0.5, color=FG.ACCENT, alpha=0.8,
                label=f"Mizan closure, 90% ({v['ETH']['mae_mcm']:.1f})")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("metered county-annual pumping, Mm$^3$/yr")
    ax.set_ylabel("estimated, Mm$^3$/yr")
    ax.set_title("Against the withheld meters, 2000 to 2024")
    ax.legend(fontsize=7.5, loc="upper left")
    FG.despine(ax)

    axes = [fig.add_subplot(gs[1, j]) for j in range(3)]
    ax = axes[0]
    ax.plot(years, q_true.sum(axis=0), color=FG.INK, lw=2.0, label="metered total")
    ax.fill_between(years, lo.sum(axis=0), hi.sum(axis=0), color=FG.ACCENT, alpha=0.20)
    ax.plot(years, hat.sum(axis=0), color=FG.ACCENT, lw=2.0, label="Mizan closure")
    ax.plot(years, ol.sum(axis=0), color=FG.WARM, lw=1.3, ls="--", label="open loop")
    ax.set_ylabel("Mm$^3$/yr")
    ax.set_title("Six-county total")
    ax.legend(fontsize=7)
    FG.despine(ax)

    ax = axes[1]
    err = np.abs(hat - q_true).mean(axis=1)
    errb = np.abs(ol - q_true).mean(axis=1)
    w = 0.38
    x = np.arange(len(K.COUNTIES))
    ax.bar(x - w / 2, errb, width=w, color=FG.WARM, label="open loop")
    ax.bar(x + w / 2, err, width=w, color=FG.ACCENT, label="closure")
    ax.set_xticks(x)
    ax.set_xticklabels([K.COUNTY_NAME[c] for c in K.COUNTIES], fontsize=7.0,
                       rotation=30, ha="right")
    ax.set_ylabel("mean absolute error, Mm$^3$/yr")
    ax.set_title("Per county")
    ax.legend(fontsize=7)
    FG.despine(ax)

    ax = axes[2]
    rows = [k for k in ("BASELINE", "BASELINE_ORACLE", "PRIOR_FLAT", "ET", "H", "ETH")
            if k in v]
    lab = {"BASELINE": "open loop\n0.80", "BASELINE_ORACLE": "open loop\nfitted",
           "PRIOR_FLAT": "area x\none acre-ft", "ET": "ET only", "H": "heads only",
           "ETH": "closure"}
    val = [v[k]["mae_mcm"] for k in rows]
    colour = {"BASELINE": FG.WARM, "BASELINE_ORACLE": FG.WARM, "PRIOR_FLAT": FG.INK,
              "ET": FG.MUTED, "H": FG.MUTED, "ETH": FG.ACCENT}
    col = [colour[k] for k in rows]
    ax.bar(range(len(rows)), val, color=col, width=0.62)
    for i, y in enumerate(val):
        ax.text(i, y * 1.02, f"{y:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([lab[k].replace("\n", " ") for k in rows], fontsize=7.0,
                       rotation=30, ha="right")
    ax.set_ylabel("MAE, Mm$^3$/yr")
    ax.set_title("Every bar a reviewer can compute, on real data")
    FG.despine(ax)

    return FG.save(fig, FIG / "fig11_kansas.png")


def main():
    tr = np.load(RES / "truth.npz")
    ab = json.loads((RES / "ablation.json").read_text())
    made = [fig_context(), fig_basin(tr), fig_ablation(ab), fig_calibration(ab),
            fig_scatter(tr), fig_closure(tr), fig_identity(), fig_nullspace()]
    if (RES / "allocation.json").exists():
        made.append(fig_allocation())
    if (RES / "voi.json").exists():
        made.append(fig_voi())
    if (RES / "detection.json").exists():
        made.append(fig_detection())
    if (RES / "kansas_v3.json").exists():
        made.append(fig_kansas())
    for p in made:
        print("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
