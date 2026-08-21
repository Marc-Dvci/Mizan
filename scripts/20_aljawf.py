"""L3 Al Jawf: what the published instruments say about the Saq, and how far apart.

L0 shows the closure is identifiable when the answer is known. L2 puts two of its legs
against real meters. Neither says anything about the basin this prize is run for, and no
metered abstraction exists there to be scored against. What can be done on Al Jawf is the
thing that motivates the whole entry: take the instruments a regulator would reach for
today, run them over the same irrigated pixels in the same years, and report how far
apart their answers are.

Two accounts, both entirely from public data:

* **The consumptive-use leg.** Centre-pivot extent from the annual maximum MODIS NDVI,
  which in a landscape receiving under 60 mm/yr of rain is an unambiguous marker of
  irrigation, and then actual evapotranspiration over exactly those pixels from every
  global product that covers them, plus a crop-coefficient account built the way the
  published study for this basin built it.

* **The gravimetric leg, with a control.** The mass trend over the Saq footprint, and the
  same trend over a Rub' al Khali box with almost no irrigation. A raw trend over a
  desert aquifer is not an abstraction signal until the regional field is removed, and
  nothing published for this basin removes it.

    python scripts/20_aljawf.py [--project EE_PROJECT]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# The Al Jawf pivot field, Wadi As-Sirhan basin. The box is drawn around the cultivated
# area south of the Jordanian border and reported with the delineation, because a
# threshold on a bigger box is a different number.
AOI = [38.2, 29.4, 40.0, 30.9]

# The Saq-Ram footprint under and around the cultivated area, and controls with almost no
# irrigation and the same satellite record. There is no single right control box, so
# several are run and the spread across them is the result rather than any one of them.
SAQ = [37.0, 27.5, 42.0, 31.5]
SAQ_TIGHT = [38.0, 29.0, 40.5, 31.2]
CONTROLS = {
    "Rub' al Khali, 47-52E 18.5-22.5N": [47.0, 18.5, 52.0, 22.5],
    "An-Nafud and eastern shield, 42-46E 27-30N": [42.0, 27.0, 46.0, 30.0],
    "central Arabian shield, 42-46E 22-26N": [42.0, 22.0, 46.0, 26.0],
    "western Rub' al Khali, 45-49E 17-21N": [45.0, 17.0, 49.0, 21.0],
}

# Annual maximum NDVI above which a pixel is called irrigated. Rainfall over the basin is
# under 60 mm/yr, so no rainfed vegetation reaches this and the threshold is not a
# tuning knob. The sensitivity across 0.35 to 0.50 is reported with the result.
NDVI_THRESHOLD = 0.40

# Published account for this basin: López Valencia et al. 2020, HESS 24, 5251-5277,
# who delineate the pivots at 30 m and report the abstraction the entry compares against.
PUBLISHED_YEAR = 2015
PUBLISHED_PIVOT_KM2 = 2494.0
PUBLISHED_ABSTRACTION_MCM = 5500.0
PUBLISHED_EFFICIENCY = 0.80

YEARS_ET = (2015, 2019, 2021)
GRACE_START, GRACE_END = "2002-04-01", "2025-01-01"


def annual_pivots(ee, year: int, thr: float = NDVI_THRESHOLD):
    """Irrigated mask and its area for one year, from the annual maximum NDVI."""
    nd = (ee.ImageCollection("MODIS/061/MOD13Q1")
          .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
          .select("NDVI").max().multiply(1e-4))
    return nd.gt(thr).selfMask(), nd


def masked_mean(ee, img, mask, scale: float) -> float:
    """Area-weighted mean of `img` over the irrigated pixels, in the image's own units."""
    v = (img.updateMask(mask).reduceRegion(
        ee.Reducer.mean(), ee.Geometry.Rectangle(AOI), scale,
        bestEffort=True, maxPixels=1e10).getInfo())
    return float(next(iter(v.values()))) if v and next(iter(v.values())) is not None else float("nan")


def area_km2(ee, mask) -> float:
    v = (mask.multiply(ee.Image.pixelArea()).reduceRegion(
        ee.Reducer.sum(), ee.Geometry.Rectangle(AOI), 250,
        bestEffort=True, maxPixels=1e10).getInfo())
    return float(next(iter(v.values()))) / 1e6


def wapor_annual(ee, year: int, version: str):
    """WaPOR actual evapotranspiration and interception, mm/yr.

    Dekadal composites carrying a daily rate at a scale factor of 0.1, so a dekad
    contributes rate times its own length rather than a fixed ten days.
    """
    coll = {"v3": ("FAO/WAPOR/3/L1_AETI_D", "L1-AETI-D"),
            "v2": ("FAO/WAPOR/2/L1_AETI_D", "L1_AETI_D")}[version]
    c = ee.ImageCollection(coll[0]).filterDate(f"{year}-01-01", f"{year + 1}-01-01")

    def scale(img):
        # Dekads start on the 1st, the 11th and the 21st, so the third one of a month
        # is eight to eleven days long and a fixed ten would misprice February by a
        # fifth. The length is taken from the image's own date.
        d0 = ee.Date(img.get("system:time_start"))
        day = ee.Number(d0.get("day"))
        last = ee.Number(d0.advance(1, "month").advance(-1, "day").get("day")).subtract(20)
        nd = ee.Number(ee.Algorithms.If(day.gte(21), last, 10))
        return img.select(coll[1]).multiply(0.1).multiply(nd)

    return c.map(scale).sum().rename("et")


PML = "projects/pml_evapotranspiration/PML/OUTPUT/PML_V22a"


def pml_annual(ee, year: int):
    """PML V2 actual evapotranspiration, mm/yr, as the sum of its three components.

    The current asset, which supersedes `CAS/IGSNRR/PML/V2_v017`; the retired one stops
    in 2020 and returns an empty collection rather than an error for any later year.
    """
    c = ee.ImageCollection(PML).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
    return c.map(lambda i: i.select(["Ec", "Es", "Ei"]).reduce(ee.Reducer.sum())
                 ).sum().rename("et")


def kc_annual(ee, year: int):
    """The crop-coefficient account, mm/yr, built the way the published study built it.

    Reference evapotranspiration from TerraClimate at 4 km, times a crop coefficient
    read off the NDVI of the same pixel, which is the standard linear relation used
    where no flux tower exists. It is the account a hydrologist without a thermal
    retrieval would write down, and it is the closest in method to the published one.
    """
    eto = (ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
           .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
           .select("pet").sum().multiply(0.1))

    def kc_of(img):
        return img.select("NDVI").multiply(1e-4).multiply(1.25).add(0.20).clamp(0.0, 1.15)

    nd = (ee.ImageCollection("MODIS/061/MOD13Q1")
          .filterDate(f"{year}-01-01", f"{year + 1}-01-01").select("NDVI"))
    kc_mean = nd.map(kc_of).mean()
    return eto.multiply(kc_mean).rename("et")


def reference_et(ee, year: int):
    """Reference evapotranspiration over the same pixels, mm/yr.

    It is the ceiling every actual-evapotranspiration account has to sit under. A
    well-watered crop can approach it and cannot exceed it by more than its crop
    coefficient, so an account above it is a unit error rather than a retrieval.
    """
    return (ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .select("pet").sum().multiply(0.1).rename("et"))


def terraclimate_aet(ee, year: int):
    """The water-balance model's own actual evapotranspiration, mm/yr."""
    return (ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .select("aet").sum().multiply(0.1).rename("et"))


def grace_series(ee, box) -> tuple[np.ndarray, np.ndarray]:
    """Monthly liquid water equivalent thickness over a box, decimal years and cm."""
    c = (ee.ImageCollection("NASA/GRACE/MASS_GRIDS_V04/MASCON")
         .filterDate(GRACE_START, GRACE_END).select("lwe_thickness"))
    geom = ee.Geometry.Rectangle(box)

    def one(img):
        v = img.reduceRegion(ee.Reducer.mean(), geom, 25000, bestEffort=True,
                             maxPixels=1e9).get("lwe_thickness")
        return ee.Feature(None, {"t": img.get("system:time_start"), "v": v})

    fc = c.map(one).filter(ee.Filter.notNull(["v"])).getInfo()["features"]
    t = np.array([f["properties"]["t"] for f in fc], dtype=float) / 1000.0
    v = np.array([f["properties"]["v"] for f in fc], dtype=float)
    return 1970.0 + t / (365.25 * 86400.0), v


def trend(t: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """Linear trend in cm per decade, with the standard error of the slope.

    An annual harmonic is carried alongside the line so that the seasonal cycle is not
    aliased into the slope by an uneven sampling record.
    """
    w = 2.0 * np.pi
    A = np.column_stack([np.ones_like(t), t - t.mean(),
                         np.cos(w * t), np.sin(w * t)])
    beta, *_ = np.linalg.lstsq(A, v, rcond=None)
    resid = v - A @ beta
    dof = max(len(t) - A.shape[1], 1)
    cov = (resid @ resid) / dof * np.linalg.inv(A.T @ A)
    return float(beta[1] * 10.0), float(np.sqrt(cov[1, 1]) * 10.0)


def box_area_km2(box) -> float:
    lon0, lat0, lon1, lat1 = box
    lat = np.deg2rad(0.5 * (lat0 + lat1))
    return (lon1 - lon0) * 111.32 * np.cos(lat) * (lat1 - lat0) * 110.574


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.environ.get("EARTHENGINE_PROJECT"),
                    help="Earth Engine cloud project, or set EARTHENGINE_PROJECT. "
                         "Registration is free and the assets used here are public.")
    args = ap.parse_args()
    if not args.project:
        raise SystemExit("pass --project or set EARTHENGINE_PROJECT: this rung reads "
                         "live from Earth Engine and needs an authenticated project")

    import ee
    ee.Initialize(project=args.project)
    out: dict = {"_aoi": AOI, "_ndvi_threshold": NDVI_THRESHOLD,
                 "_published": {"source": "López Valencia et al. 2020, HESS 24, 5251-5277",
                                "year": PUBLISHED_YEAR,
                                "pivot_km2": PUBLISHED_PIVOT_KM2,
                                "abstraction_mcm": PUBLISHED_ABSTRACTION_MCM}}

    # ---------------------------------------------------------------- pivot extent
    print("centre-pivot extent from the annual maximum NDVI\n")
    sens = {}
    for thr in (0.35, 0.40, 0.45, 0.50):
        m, _ = annual_pivots(ee, PUBLISHED_YEAR, thr)
        sens[f"{thr:.2f}"] = area_km2(ee, m)
        print(f"  threshold {thr:.2f}  {PUBLISHED_YEAR}: {sens[f'{thr:.2f}']:7,.0f} km2")
    out["_threshold_sensitivity_km2"] = sens
    print(f"  published, delineated at 30 m: {PUBLISHED_PIVOT_KM2:7,.0f} km2\n")

    masks, areas = {}, {}
    for y in YEARS_ET:
        masks[y], _ = annual_pivots(ee, y)
        areas[y] = area_km2(ee, masks[y])
        print(f"  {y}: {areas[y]:,.0f} km2 irrigated")
    out["pivot_km2"] = areas

    # ------------------------------------------------------- the consumptive-use leg
    print("\nactual evapotranspiration over exactly those pixels, mm/yr\n")
    print(f"{'account':44s} " + " ".join(f"{y:>10d}" for y in YEARS_ET))
    accounts = {
        "WaPOR v3, 326 m": (lambda y: wapor_annual(ee, y, "v3"), 326, 2018),
        "WaPOR v2, 250 m": (lambda y: wapor_annual(ee, y, "v2"), 250, 2009),
        "PML V2, 500 m": (lambda y: pml_annual(ee, y), 500, 2000),
        "TerraClimate water balance, 4 km": (lambda y: terraclimate_aet(ee, y), 4638, 1958),
        "crop coefficient from NDVI x reference ET": (lambda y: kc_annual(ee, y), 250, 2000),
    }
    et = {}
    for name, (fn, scale, first) in accounts.items():
        row = {}
        for y in YEARS_ET:
            row[y] = (masked_mean(ee, fn(y), masks[y], scale)
                      if y >= first else float("nan"))
        et[name] = row
        print(f"{name:44s} " + " ".join(
            ("       n/a" if not np.isfinite(row[y]) else f"{row[y]:10.0f}")
            for y in YEARS_ET))
    out["et_mm_yr"] = {k: {str(y): v for y, v in r.items()} for k, r in et.items()}

    eto = {y: masked_mean(ee, reference_et(ee, y), masks[y], 4638) for y in YEARS_ET}
    out["reference_et_mm_yr"] = {str(y): v for y, v in eto.items()}
    print(f"{'reference evapotranspiration, the ceiling':44s} " + " ".join(
        f"{eto[y]:10.0f}" for y in YEARS_ET))

    # Every retrieval has to sit under the reference. PML's eight-day composites carry
    # the period total rather than a daily rate, and reading them as a rate would put
    # the account six times above the ceiling. This is the check that settles it.
    CEIL = 1.20
    for name, row in et.items():
        for y, v in row.items():
            if np.isfinite(v) and v > CEIL * eto[y]:
                raise SystemExit(
                    f"{name} reports {v:,.0f} mm/yr in {y} against a reference "
                    f"evapotranspiration of {eto[y]:,.0f}. That is a unit error.")

    # TerraClimate is reported apart from the rest: it is a water-balance model with no
    # irrigation term, so it is not a retrieval that disagrees, it is an instrument that
    # cannot see the agriculture at all.
    fin = {k: r[PUBLISHED_YEAR] for k, r in et.items()
           if np.isfinite(r[PUBLISHED_YEAR]) and not k.startswith("TerraClimate")}
    spread = max(fin.values()) / min(fin.values())
    out["et_spread_factor"] = spread
    print(f"\nOver the same pixels in {PUBLISHED_YEAR} the retrievals span a factor of "
          f"{spread:.1f}, from {min(fin.values()):,.0f} to {max(fin.values()):,.0f} mm/yr "
          f"under a reference of {eto[PUBLISHED_YEAR]:,.0f}.")
    tcv = et["TerraClimate water balance, 4 km"][PUBLISHED_YEAR]
    print(f"The global water-balance model reports {tcv:,.0f} mm/yr over the same "
          f"pixels, {tcv / eto[PUBLISHED_YEAR] * 100:.0f} per cent of the reference: it "
          f"has no irrigation term and does not see the agriculture.")

    print(f"\nabstraction implied at an efficiency of {PUBLISHED_EFFICIENCY:.2f}, Mm3/yr\n")
    abst = {}
    for name, row in et.items():
        v = row[PUBLISHED_YEAR]
        if np.isfinite(v):
            abst[name] = v * 1e-3 * areas[PUBLISHED_YEAR] * 1e6 / PUBLISHED_EFFICIENCY / 1e6
            print(f"{name:44s} {abst[name]:10,.0f}")
    print(f"{'published for this basin':44s} {PUBLISHED_ABSTRACTION_MCM:10,.0f}")
    out["abstraction_mcm"] = abst

    # -------------------------------------------------------- the gravimetric leg
    print("\nbasin mass trend, and the same trend over deserts with no irrigation\n")
    ts, vs = grace_series(ee, SAQ)
    b_s, se_s = trend(ts, vs)
    tt, vt = grace_series(ee, SAQ_TIGHT)
    b_t, se_t = trend(tt, vt)
    a_saq = box_area_km2(SAQ)
    print(f"  {'Saq footprint':44s} {b_s:+7.2f} +/- {se_s:.2f} cm/decade   "
          f"{len(ts)} months, {ts.min():.1f} to {ts.max():.1f}")
    print(f"  {'Saq, tight box on the cultivated area':44s} {b_t:+7.2f} +/- "
          f"{se_t:.2f} cm/decade\n")

    ctrl = {}
    for name, box in CONTROLS.items():
        tc, vc = grace_series(ee, box)
        b_c, se_c = trend(tc, vc)
        share = (b_s - b_c) / b_s * 100.0
        vol = -(b_s - b_c) * 1e-3 * a_saq
        ctrl[name] = {"cm_decade": b_c, "se": se_c, "box": box,
                      "differenced_cm_decade": b_s - b_c,
                      "local_share_pct": share, "differenced_mcm_yr": vol,
                      "series": {"t": tc.tolist(), "v": vc.tolist()}}
        print(f"  {name:44s} {b_c:+7.2f} +/- {se_c:.2f}   differenced "
              f"{b_s - b_c:+6.2f}   local {share:3.0f}%   {vol:6,.0f} Mm3/yr")

    lo = min(c["local_share_pct"] for c in ctrl.values())
    hi = max(c["local_share_pct"] for c in ctrl.values())
    vlo = min(c["differenced_mcm_yr"] for c in ctrl.values())
    vhi = max(c["differenced_mcm_yr"] for c in ctrl.values())
    kc_vol = (et["crop coefficient from NDVI x reference ET"][PUBLISHED_YEAR]
              * 1e-3 * areas[PUBLISHED_YEAR])
    print(f"\n  the local share of the raw trend is between {lo:.0f} and {hi:.0f} "
          f"per cent depending on which desert is the control")
    print(f"  differenced storage loss over {a_saq:,.0f} km2: "
          f"{vlo:,.0f} to {vhi:,.0f} Mm3/yr")
    print(f"  crop-coefficient consumptive use over the pivots, {PUBLISHED_YEAR}: "
          f"{kc_vol:,.0f} Mm3/yr")
    print(f"  the two satellite legs disagree by a factor of "
          f"{kc_vol / vhi:.1f} to {kc_vol / vlo:.1f}")

    out["grace"] = {"saq_cm_decade": b_s, "saq_se": se_s, "saq_box": SAQ,
                    "saq_tight_cm_decade": b_t, "saq_tight_se": se_t,
                    "saq_tight_box": SAQ_TIGHT,
                    "saq_area_km2": a_saq,
                    "controls": ctrl,
                    "local_share_pct_range": [lo, hi],
                    "differenced_mcm_yr_range": [vlo, vhi],
                    "kc_consumptive_mcm_yr": kc_vol,
                    "n_months": int(len(ts)),
                    "span": [float(ts.min()), float(ts.max())],
                    "series": {"t_saq": ts.tolist(), "v_saq": vs.tolist(),
                               "t_saq_tight": tt.tolist(), "v_saq_tight": vt.tolist()}}

    RES.mkdir(exist_ok=True)
    (RES / "aljawf.json").write_text(json.dumps(out, indent=2))
    print("\nwrote results/aljawf.json")


if __name__ == "__main__":
    main()
