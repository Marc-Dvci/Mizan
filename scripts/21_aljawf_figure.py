"""The Al Jawf figure: two published instruments, one aquifer, and how far apart they are.

Left: actual evapotranspiration over exactly the same irrigated pixels in the same years,
from every global product that covers them, against reference evapotranspiration as a
climatic benchmark and against the published account for this basin.

Right: the gravimetric record over the Saq footprint and over four deserts with no
irrigation, which is what a raw trend has to be read against before any of it is called
abstraction.

    python scripts/21_aljawf_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt

from mizan import figures as FG

RES = ROOT / "results"
FIG = ROOT / "figures"

ORDER = ["TerraClimate water balance, 4 km", "WaPOR v3, 326 m", "WaPOR v2, 250 m",
         "crop coefficient from NDVI x reference ET", "PML V2, 500 m"]
SHORT = {
    "WaPOR v3, 326 m": "WaPOR v3, 326 m",
    "WaPOR v2, 250 m": "WaPOR v2, 250 m",
    "PML V2, 500 m": "PML V2, 500 m",
    "TerraClimate water balance, 4 km": "TerraClimate, 4 km",
    "crop coefficient from NDVI x reference ET": "crop coefficient\nfrom NDVI",
}


def main() -> None:
    d = json.loads((RES / "aljawf.json").read_text())
    et = d["et_mm_yr"]
    ya, yb = "2015", "2021"
    eto = d["reference_et_mm_yr"]

    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.2))

    # ------------------------------------------------------------------ left panel
    a = ax[0]
    y = np.arange(len(ORDER))
    h = 0.34
    for k, (yr, col, lab) in enumerate(((ya, FG.MUTED, ya), (yb, FG.ACCENT, yb))):
        v = [et[n].get(yr, float("nan")) for n in ORDER]
        v = [0.0 if not np.isfinite(x) else x for x in v]
        a.barh(y + (0.5 - k) * h, v, height=h, color=col, label=lab)
        for i, val in enumerate(v):
            txt = "not published" if val == 0 else f"{val:,.0f}"
            a.text(max(val, 0) + 40, y[i] + (0.5 - k) * h, txt,
                   va="center", fontsize=7.2, color=FG.INK)

    a.axvline(eto[ya], color=FG.INK, lw=1.0, ls="--")
    a.text(eto[ya] - 60, len(ORDER) - 1.55, "reference\nevapotranspiration",
           fontsize=7.2, ha="right", va="center", color=FG.INK)

    pub = d["_published"]["abstraction_mcm"] * 1e3 * 0.80 / d["pivot_km2"][ya]
    a.axvline(pub, color=FG.GREEN, lw=1.4)
    a.text(pub - 60, 0.55, "published for\nthis basin, 2015",
           fontsize=7.2, ha="right", va="center", color=FG.GREEN)

    a.set_yticks(y)
    a.set_yticklabels([SHORT[n] for n in ORDER], fontsize=8)
    a.set_xlim(0, eto[ya] * 1.30)
    a.set_xlabel("actual evapotranspiration over the irrigated pixels, mm/yr")
    a.set_title(f"Five instruments over the same {d['pivot_km2'][ya]:,.0f} km$^2$"
                " of pivots", fontsize=9.5)
    a.legend(loc="lower right", fontsize=8)
    a.grid(axis="y", visible=False)
    FG.despine(a)

    # ----------------------------------------------------------------- right panel
    b = ax[1]
    g = d["grace"]
    t = np.array(g["series"]["t_saq"])
    v = np.array(g["series"]["v_saq"])

    ctrl = sorted(g["controls"].items(), key=lambda kv: kv[1]["cm_decade"])
    for name, c in ctrl:
        tc, vc = np.array(c["series"]["t"]), np.array(c["series"]["v"])
        b.plot(tc, vc - vc[:12].mean(), color=FG.MUTED, lw=0.7, alpha=0.8, zorder=1)
    b.plot(t, v - v[:12].mean(), color=FG.WARM, lw=1.2, zorder=3,
           label=f"Saq footprint   {g['saq_cm_decade']:+.1f} cm/decade")
    b.plot([], [], color=FG.MUTED, lw=0.7,
           label=f"four deserts with no irrigation   {ctrl[0][1]['cm_decade']:+.1f} "
                 f"to {ctrl[-1][1]['cm_decade']:+.1f}")
    b.legend(loc="upper right", fontsize=7.8)

    lo, hi = g["local_share_pct_range"]
    vlo, vhi = g["differenced_mcm_yr_range"]
    b.text(0.02, 0.02,
           f"Subtract a control and {lo:.0f} to {hi:.0f} per cent\n"
           f"of the trend is left as local:\n"
           f"{vlo:,.0f} to {vhi:,.0f} Mm$^3$/yr, a factor of {vhi / vlo:.0f},\n"
           f"from a choice nobody publishes.",
           transform=b.transAxes, ha="left", va="bottom", fontsize=8.0, color=FG.INK)

    b.set_xlabel("year")
    b.set_ylabel("liquid water equivalent, cm, against the first year")
    b.set_title("The trend, and what four deserts with no wells do over the same years",
                fontsize=9.5)
    FG.despine(b)

    fig.tight_layout()
    out = FG.save(fig, FIG / "fig10_aljawf.png")
    print("wrote", out)


if __name__ == "__main__":
    main()
