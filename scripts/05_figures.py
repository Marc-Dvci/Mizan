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

    fig, ax = plt.subplots(figsize=(8.2, 4.1))
    b = ax.bar(range(len(rows)), mae, color=col, width=0.68)
    ax.axhline(base, color=FG.WARM, lw=1.6,
               label=f"published open loop, fixed 0.80 ({base:.1f})")
    ax.axhline(oracle, color=FG.WARM, lw=1.2, ls="--",
               label=f"open loop, efficiency fitted to the answer ({oracle:.1f})")
    for r, v in zip(b, mae):
        ax.text(r.get_x() + r.get_width() / 2, v * 1.04, f"{v:.1f}", ha="center",
                va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(4, 260)
    ax.set_yticks([5, 10, 20, 50, 100, 200])
    ax.get_yaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([SHORT[k] for k in rows], rotation=32, ha="right")
    ax.set_ylabel("district-annual abstraction MAE, Mm$^3$/yr (log)")
    ax.set_title("What each observation is worth, against withheld truth")
    ax.legend(loc="upper right", fontsize=8)
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
    for k, x, y in zip(rows, w, m):
        ax[1].annotate(SHORT[k], (x, y), fontsize=7, xytext=(3, 3),
                       textcoords="offset points")
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
    order_m = [k for k in v["fc_q_last5"]["greedy_order"] if k.startswith("meter")]
    rank = {int(k.split("_d")[1]): i + 1 for i, k in enumerate(order_m)}
    for dd, r in rank.items():
        i, j = divmod(dd, 3)
        ax[1].text((j + 0.5) * third, (i + 0.72) * third, f"meter {r}",
                   ha="center", va="center", fontsize=8.5, color=FG.WARM, weight="bold")
    for k in (1, 2):
        ax[1].axhline(k * third, color=FG.INK, lw=0.6, alpha=0.5)
        ax[1].axvline(k * third, color=FG.INK, lw=0.6, alpha=0.5)

    geo = [k for k in v["fc_perm_loss"]["greedy_order"] if not k.startswith("meter")]
    for r, n in enumerate(geo[:8]):
        kind, j = n.rsplit("_", 1)
        rr, cl = cc[int(j)]
        x = (cl + 0.5) * C.EST.delr_m / 1000.0
        y = (rr + 0.5) * C.EST.delr_m / 1000.0
        ax[1].scatter([x], [y], s=150, marker="o" if kind == "piezo" else "^",
                      facecolor="white", edgecolor=FG.ACCENT, lw=1.5, zorder=3)
        ax[1].text(x, y, str(r + 1), fontsize=7, ha="center", va="center", zorder=4)
    ax[1].set_title("Where the next instrument goes: meter rank for\n"
                    "abstraction, geodesy for capacity")
    ax[1].set_xlabel("km")
    ax[1].set_ylabel("km")
    ax[1].grid(False)
    fig.tight_layout()
    return FG.save(fig, FIG / "fig8_voi.png")


def main():
    tr = np.load(RES / "truth.npz")
    ab = json.loads((RES / "ablation.json").read_text())
    made = [fig_basin(tr), fig_ablation(ab), fig_calibration(ab), fig_scatter(tr),
            fig_closure(tr), fig_nullspace()]
    if (RES / "allocation.json").exists():
        made.append(fig_allocation())
    if (RES / "voi.json").exists():
        made.append(fig_voi())
    for p in made:
        print("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
