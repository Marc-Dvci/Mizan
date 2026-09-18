"""Why the synthetic error rate does not transfer: the two basins are not the same problem.

The L0 twin is a fossil basin. Over its twenty years it abstracts 31.04 km3 and depletes
29.06 km3 of storage, so 6 per cent of what is pumped is replaced by inflow and 94 per
cent comes out of storage. Northwest Kansas is not that basin, and the difference is
measurable from the same public records the rung already uses, with no meter and no
model:

    N = Q + Sy * A * dh        (m3/yr, dh negative where the water table falls)

with `Q` the district's pumping, `dh` the winter-to-winter water-table change averaged
over the wells of that county, `Sy` the USGS specific-yield map and `A` the county area.
`N` is net inflow, the quantity Butler et al. (2016, 2020, 2023) target for
sustainability in exactly these counties, and `N/Q` is the share of pumping the aquifer
replaces within the year.

This is a diagnostic, not a score. It is computed from the withheld meters, so it is
read once, at the end, like every other scored quantity, and it is reported because it
sizes the distance between the two rungs rather than asserting it.

    python scripts/28_net_inflow.py

Writes results/net_inflow.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mizan import ks_data as K

RES = ROOT / "results"

# The L0 twin, from results/truth.npz and reported in RESULTS.md.
TWIN_ABSTRACTED_KM3 = 31.04
TWIN_DEPLETED_KM3 = 29.06


def main() -> None:
    region = K.build_region(K.load_points())
    wl = K.water_levels(region)
    years = np.arange(K.YEAR0, K.YEAR1 + 1)
    q_by, _, _ = K.reported_annual()
    q = q_by.sum(axis=0)

    sy = K.usgs_field(region, "sy")
    heads = np.array([w["head"] for w in wl["wells"]])
    cty = np.array([region.county[w["row"], w["col"]] for w in wl["wells"]])

    nc = len(K.COUNTIES)
    dwl = np.full((nc, years.size - 1), np.nan)
    nwell = np.zeros((nc, years.size - 1), dtype=int)
    for i in range(nc):
        h = heads[cty == i]
        d = h[:, 1:] - h[:, :-1]
        dwl[i] = np.nanmean(d, axis=0)
        nwell[i] = np.sum(np.isfinite(d), axis=0)

    area = np.array([(region.county == i).sum() * region.area_m2 for i in range(nc)])
    sy_c = np.array([float(sy[region.county == i].mean()) for i in range(nc)])

    # Year t's pumping against the level change from winter t to winter t+1.
    qq = q[:, :-1]
    dS = sy_c[:, None] * area[:, None] * dwl
    N = qq + dS

    era = years[:-1] >= K.METERED_ERA_YEAR0
    out = {
        "_note": ("net inflow N = Q + Sy*A*dh from the WIMAS meters, the WIZARD winter "
                  "levels and the USGS specific-yield map; N/Q is the share of pumping "
                  "the aquifer replaces within the year"),
        "_metered_era": int(K.METERED_ERA_YEAR0),
        "_specific_yield_by_county": dict(zip(K.COUNTIES, sy_c.round(4).tolist())),
        "_wells_per_county_year_median": int(np.median(nwell)),
        "by_county": {}, "block": {}, "twin": {},
    }
    for i, c in enumerate(K.COUNTIES):
        out["by_county"][c] = {
            "pumping_mcm_yr": float(qq[i, era].mean() / 1e6),
            "water_level_change_m_yr": float(dwl[i, era].mean()),
            "net_inflow_mcm_yr": float(N[i, era].mean() / 1e6),
            "net_inflow_over_pumping": float(N[i, era].mean() / qq[i, era].mean()),
        }
    nq_year = N.sum(axis=0) / qq.sum(axis=0)
    out["block"] = {
        "pumping_mcm_yr": float(qq[:, era].sum(axis=0).mean() / 1e6),
        "net_inflow_mcm_yr": float(N[:, era].sum(axis=0).mean() / 1e6),
        "net_inflow_over_pumping": float(N[:, era].sum(axis=0).mean()
                                         / qq[:, era].sum(axis=0).mean()),
        "net_inflow_over_pumping_by_year": {int(y): float(v) for y, v
                                            in zip(years[:-1], nq_year)},
        "net_inflow_over_pumping_year_sd": float(np.std(nq_year[era])),
        "storage_share_of_pumping": float(
            1.0 - N[:, era].sum(axis=0).mean() / qq[:, era].sum(axis=0).mean()),
    }
    out["twin"] = {
        "abstracted_km3": TWIN_ABSTRACTED_KM3,
        "storage_depleted_km3": TWIN_DEPLETED_KM3,
        "net_inflow_over_pumping": float(
            (TWIN_ABSTRACTED_KM3 - TWIN_DEPLETED_KM3) / TWIN_ABSTRACTED_KM3),
        "storage_share_of_pumping": float(TWIN_DEPLETED_KM3 / TWIN_ABSTRACTED_KM3),
    }
    out["ratio_of_storage_shares"] = (out["twin"]["storage_share_of_pumping"]
                                      / out["block"]["storage_share_of_pumping"])
    (RES / "net_inflow.json").write_text(json.dumps(out, indent=2))

    print("Net inflow over the metered era, {} to {}\n".format(
        K.METERED_ERA_YEAR0, K.YEAR1 - 1))
    print("{:<10s} {:>10s} {:>10s} {:>12s} {:>7s}".format(
        "county", "Q Mm3/yr", "dWL m/yr", "N Mm3/yr", "N/Q"))
    for c in K.COUNTIES:
        v = out["by_county"][c]
        print("{:<10s} {:>10.1f} {:>+10.2f} {:>12.1f} {:>7.2f}".format(
            K.COUNTY_NAME[c], v["pumping_mcm_yr"], v["water_level_change_m_yr"],
            v["net_inflow_mcm_yr"], v["net_inflow_over_pumping"]))
    b = out["block"]
    print("{:<10s} {:>10.1f} {:>10s} {:>12.1f} {:>7.2f}".format(
        "BLOCK", b["pumping_mcm_yr"], "", b["net_inflow_mcm_yr"],
        b["net_inflow_over_pumping"]))
    print("\nStorage supplies {:.0f} per cent of what Northwest Kansas pumps and {:.0f} "
          "per cent of what\nthe twin pumps: the twin is {:.1f} times more "
          "storage-driven. The closure reads storage,\nso the two rungs are not the same "
          "problem and the twin's error rate is not a\nprediction for a fossil basin; "
          "the Saq, with recharge near zero, sits at the twin's end\nof that range and "
          "not at Kansas's.".format(
              100 * b["storage_share_of_pumping"],
              100 * out["twin"]["storage_share_of_pumping"],
              out["ratio_of_storage_shares"]))
    print("\nwrote results/net_inflow.json")


if __name__ == "__main__":
    main()
