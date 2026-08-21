# -*- coding: utf-8 -*-
"""Compute the mascon gain for the Saq rather than assuming one.

The gravity operator reads

    G(t) = alpha * dS(t) / A + external(t) + eps

and `alpha`, the mascon gain, multiplies the quantity the gravity leg exists to supply.
Script 23 puts the width of the prior on `alpha` on an axis and scores what it costs.
This script answers the prior question, and answers it for the basin the prize is run
for: **what is `alpha` on the Saq, computed rather than assumed?**

The gain is not a property of the instrument. It is a property of three things together:
the mascon tessellation, the footprint the account is reported over, and the spatial
extent of the storage change itself. All three are public or observable, so the gain is
computable, and this script computes it.

**The tessellation is recovered from the product.** A JPL mascon solution is piecewise
constant on its own mascons, so sampling the published grid at its 0.5 degree posting and
grouping cells that carry identical values at several independent epochs returns the
mascon polygons themselves. No placement file and no assumption about the tessellation is
needed, and the recovered blocks are checked against the 3 degree equal-area design.

**The source is the irrigated area.** Centre-pivot extent over the Wadi As-Sirhan basin
from the annual maximum MODIS NDVI, the same delineation L3 uses, carrying a unit
depletion. Because drawdown spreads beyond the pumped area, the source is also run
spread over a sequence of radii, and the spread is reported as an axis rather than
chosen.

**The gain then follows by forward modelling.** Each mascon reports the area-weighted
mean of the true field over its own polygon. The gain over a reporting footprint is the
mass that survives that averaging inside the footprint, divided by the mass truly there.

Two results come out of it. The first is that a footprint tighter than the mascons
attenuates the gravity leg by a factor this script computes, which is why the absolute
scale cannot be read off a pivot-scale box. The second is a design rule: report the
gravity leg over a union of whole mascons that contains the depletion, and the gain is
one by construction rather than a fitted nuisance parameter.

    python scripts/24_saq_gain.py --project EE_PROJECT

Writes results/saq_gain.json and figures/fig14_saq_gain.png.
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
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mizan import figures as F

RES = ROOT / "results"
FIG = ROOT / "figures"

R_EARTH_KM = 6371.0088

# The window the tessellation is recovered over. It is deliberately much larger than any
# reporting footprint, so that every mascon touching a footprint is seen whole.
WIN = (31.0, 20.0, 50.0, 39.0)          # lon0, lat0, lon1, lat1
CELL = 0.25                              # working grid, half the mascon posting

# The window the irrigated area is measured over. Everything outside it carries no
# source mass, which is checked below by measuring the pivot area in a collar around it.
SRC = (36.5, 28.0, 41.5, 32.0)
COLLAR = (35.5, 27.0, 42.5, 33.0)

# Epochs used to recover the tessellation. Three widely separated months, so that two
# mascons sharing a value by coincidence in one month are not merged.
EPOCHS = (10, 100, 180)

# The threshold L3 uses, reported with it: rainfall over the pivot fields is under
# 60 mm/yr, so nothing rainfed reaches it there. This script works over a window several
# times wider than those fields, and over that window a single wet year does put rainfed
# and wadi vegetation above the same threshold: 2019 returns 5,389 km2 against 3,362 in
# 2021, while the published pivot box moves by under 8 per cent between the same years.
# A centre pivot is perennial, so the source is the area that clears the threshold in
# every sampled year, and the single years are kept below as the sensitivity that shows
# what a one-year delineation would have done.
NDVI_THRESHOLD = 0.40
YEARS = (2015, 2019, 2021)
SENSITIVITY = [(YEARS, 0.35), (YEARS, 0.50), ((2015,), 0.40), ((2019,), 0.40),
               ((2021,), 0.40)]

# Reporting footprints, using the same boxes L3 reports over.
FOOTPRINTS = {
    "the Al Jawf pivot box": (38.2, 29.4, 40.0, 30.9),
    "the tight Saq box": (38.0, 29.0, 40.5, 31.2),
    "the Saq box": (37.0, 27.5, 42.0, 31.5),
}

# Drawdown spreads beyond the pumped area, so the source extent is an axis, not a choice.
SPREAD_KM = (0.0, 25.0, 50.0, 100.0, 150.0)

# The prior the L0 twin ships, from src/mizan/estimator.py.
L0_PRIOR = (0.85, 0.04)


# --------------------------------------------------------------------------- geometry
def cell_grid(win, cell):
    """Cell edges and centres of a regular lon/lat grid, and each cell's area in km2."""
    lon = np.arange(win[0], win[2] + 1e-9, cell)
    lat = np.arange(win[1], win[3] + 1e-9, cell)
    clon = 0.5 * (lon[:-1] + lon[1:])
    clat = 0.5 * (lat[:-1] + lat[1:])
    # Exact area of a lon/lat cell on a sphere, so that no cosine approximation enters.
    band = (R_EARTH_KM ** 2 * np.radians(cell)
            * (np.sin(np.radians(lat[1:])) - np.sin(np.radians(lat[:-1]))))
    area = np.repeat(band[:, None], clon.size, axis=1)      # (nlat, nlon)
    return lon, lat, clon, clat, area


def box_weight(lon, lat, box):
    """Area of each cell that falls inside `box`, in km2. Exact, partial cells included."""
    lo = np.clip(lon[1:], box[0], box[2]) - np.clip(lon[:-1], box[0], box[2])
    s1 = np.sin(np.radians(np.clip(lat[1:], box[1], box[3])))
    s0 = np.sin(np.radians(np.clip(lat[:-1], box[1], box[3])))
    return R_EARTH_KM ** 2 * np.radians(lo)[None, :] * (s1 - s0)[:, None]


# ------------------------------------------------------------------------ earth engine
def mascon_labels(ee, win, cell):
    """Recover the mascon polygons from the published grid, on the working grid.

    The product is piecewise constant on its mascons, so cells carrying the same value at
    every sampled epoch belong to the same mascon. Connected components separate two
    mascons that happen to share a value in different places.
    """
    coll = ee.ImageCollection("NASA/GRACE/MASS_GRIDS_V04/MASCON").select("lwe_thickness")
    reg = ee.Geometry.Rectangle(list(win), None, False)
    stack = []
    for k in EPOCHS:
        im = ee.Image(coll.toList(1, k).get(0))
        d = im.sampleRectangle(region=reg, defaultValue=-9999).get("lwe_thickness")
        stack.append(np.array(d.getInfo(), float))
    A = np.stack(stack)                                    # (nep, nlat, nlon), row 0 north
    A = A[:, ::-1, :]                                      # south-up, to match the grid
    key = np.round(A.reshape(len(EPOCHS), -1).T, 4)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    lab = inv.reshape(A.shape[1:])
    lab = connected(lab)
    return np.kron(lab, np.ones((int(0.5 / cell), int(0.5 / cell)), int))


def connected(lab):
    """Split value-identical regions that are not contiguous, four-connectivity."""
    out = -np.ones_like(lab)
    nxt = 0
    for i in range(lab.shape[0]):
        for j in range(lab.shape[1]):
            if out[i, j] >= 0:
                continue
            v, stack = lab[i, j], [(i, j)]
            out[i, j] = nxt
            while stack:
                a, b = stack.pop()
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    p, q = a + da, b + db
                    if (0 <= p < lab.shape[0] and 0 <= q < lab.shape[1]
                            and out[p, q] < 0 and lab[p, q] == v):
                        out[p, q] = nxt
                        stack.append((p, q))
            nxt += 1
    return out


def pivot_area(ee, win, cell, years, thr):
    """Irrigated area inside every cell of `win`, km2, from the annual maximum NDVI.

    A pixel counts as irrigated only where it clears the threshold in every year of
    `years`, so that one wet season cannot enrol a wadi as a pivot field.
    """
    mask = None
    for year in years:
        nd = (ee.ImageCollection("MODIS/061/MOD13Q1")
              .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
              .select("NDVI").max().multiply(1e-4))
        m = nd.gt(thr)
        mask = m if mask is None else mask.And(m)
    img = ee.Image.pixelArea().updateMask(mask)
    lon = np.arange(win[0], win[2] - 1e-9, cell)
    lat = np.arange(win[1], win[3] - 1e-9, cell)
    feats = []
    for la in lat:
        for lo in lon:
            feats.append(ee.Feature(
                ee.Geometry.Rectangle([lo, la, lo + cell, la + cell], None, False),
                {"i": int(round((la - win[1]) / cell)),
                 "j": int(round((lo - win[0]) / cell))}))
    out = np.zeros((lat.size, lon.size))
    step = 200
    for s in range(0, len(feats), step):
        fc = ee.FeatureCollection(feats[s:s + step])
        r = img.reduceRegions(collection=fc, reducer=ee.Reducer.sum(), scale=231)
        for f in r.getInfo()["features"]:
            p = f["properties"]
            out[p["i"], p["j"]] = (p.get("sum") or 0.0) / 1.0e6
    return out


def mascon_noise(ee, box):
    """The uncertainty the product itself publishes over a footprint, mm."""
    coll = ee.ImageCollection("NASA/GRACE/MASS_GRIDS_V04/MASCON").select("uncertainty")
    g = ee.Geometry.Rectangle(list(box), None, False)
    v = coll.mean().reduceRegion(ee.Reducer.mean(), g, 25000, maxPixels=1e9).getInfo()
    u = v.get("uncertainty")
    return None if u is None else float(u) * 10.0      # cm to mm


# ---------------------------------------------------------------------------- forward
def disc(radius_km, cell, clat):
    """Mass-conserving disc kernel of the given radius on the working grid."""
    if radius_km <= 0:
        return None
    km_lat = np.radians(cell) * R_EARTH_KM
    km_lon = km_lat * np.cos(np.radians(clat.mean()))
    ri = int(np.ceil(radius_km / km_lat))
    rj = int(np.ceil(radius_km / km_lon))
    di = np.arange(-ri, ri + 1)[:, None] * km_lat
    dj = np.arange(-rj, rj + 1)[None, :] * km_lon
    k = ((di ** 2 + dj ** 2) <= radius_km ** 2).astype(float)
    return k / k.sum()


def spread(mass, kern):
    """Convolve a mass field with a kernel, conserving total mass."""
    if kern is None:
        return mass.copy()
    out = np.zeros_like(mass)
    ri, rj = kern.shape[0] // 2, kern.shape[1] // 2
    pad = np.pad(mass, ((ri, ri), (rj, rj)))
    for a in range(kern.shape[0]):
        for b in range(kern.shape[1]):
            if kern[a, b]:
                out += kern[a, b] * pad[a:a + mass.shape[0], b:b + mass.shape[1]]
    return out


def mascon_field(mass, lab, area):
    """The depth each mascon reports: its own mass spread over its own polygon."""
    out = np.zeros_like(mass)
    for m in np.unique(lab):
        s = lab == m
        a = area[s].sum()
        if a > 0:
            out[s] = mass[s].sum() / a
    return out


# ------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.environ.get("EARTHENGINE_PROJECT"),
                    help="Earth Engine cloud project, or set EARTHENGINE_PROJECT")
    ap.add_argument("--out", default="saq_gain.json")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse the cached Earth Engine reads in results/saq_gain.npz "
                         "instead of querying again; the cache records the window, the "
                         "grid and the delineation it was built under and is refused if "
                         "any of them has moved")
    ap.add_argument("--fig", default="fig14_saq_gain.png")
    args = ap.parse_args()
    cache = RES / "saq_gain.npz"
    stamp = np.array([*WIN, CELL, NDVI_THRESHOLD, *YEARS, *SRC], float)
    cached = None
    if args.reuse and cache.exists():
        z = np.load(cache, allow_pickle=True)
        if z["stamp"].shape == stamp.shape and np.allclose(z["stamp"], stamp):
            cached = z
        else:
            print("cache was built under a different configuration, re-reading")

    ee = None
    if cached is None:
        if not args.project:
            raise SystemExit(
                "pass --project or set EARTHENGINE_PROJECT: without a usable cache this "
                "script reads live from Earth Engine and needs an authenticated project")
        import ee as _ee
        ee = _ee
        ee.Initialize(project=args.project)

    lon, lat, clon, clat, area = cell_grid(WIN, CELL)
    lab = cached["lab"] if cached is not None else mascon_labels(ee, WIN, CELL)
    assert lab.shape == area.shape, (lab.shape, area.shape)

    # The recovered tessellation, checked against the published design rather than
    # trusted. Mascons touching the window edge are truncated and are reported as such.
    inner = set()
    edge = set()
    for m in np.unique(lab):
        s = lab == m
        touches = (s[0].any() or s[-1].any() or s[:, 0].any() or s[:, -1].any())
        (edge if touches else inner).add(int(m))
    sizes = {int(m): float(area[lab == m].sum()) for m in inner}
    # Two neighbouring mascons that happened to share a value at every sampled epoch
    # would have been merged into one polygon of roughly twice the area. Nothing in the
    # window is anywhere near that, which is the check that the grouping did not
    # over-merge.
    _eq3 = (np.radians(3.0) * R_EARTH_KM) ** 2
    assert max(sizes.values()) < 1.5 * _eq3, sorted(sizes.values())[-3:]
    res = {
        "_what": ("the mascon gain for the Saq, computed from the recovered mascon "
                  "geometry, the irrigated footprint and an explicit source extent"),
        "_window": list(WIN), "_cell_deg": CELL,
        "_tessellation": {
            "mascons_in_window": int(np.unique(lab).size),
            "whole_mascons": len(inner),
            "median_area_km2": float(np.median(list(sizes.values()))) if sizes else None,
            "equal_area_3deg_km2": float(
                (np.radians(3.0) * R_EARTH_KM) ** 2),
            "recovered_from": ("the product is piecewise constant on its own mascons; "
                               f"cells sharing a value at {len(EPOCHS)} epochs are one "
                               "mascon"),
        },
    }

    # ------------------------------------------------------------------ the source
    j0 = int(round((SRC[0] - WIN[0]) / CELL))
    i0 = int(round((SRC[1] - WIN[1]) / CELL))

    fields = {} if cached is None else {
        k: v for k, v in zip(cached["field_keys"], cached["field_vals"])}

    def source_field(years, thr):
        key = "|".join(str(y) for y in years) + "@{:.2f}".format(thr)
        if key not in fields:
            if ee is None:
                raise SystemExit("the cache does not carry " + key)
            fields[key] = pivot_area(ee, SRC, CELL, years, thr)
        p = fields[key]
        m = np.zeros_like(area)
        m[i0:i0 + p.shape[0], j0:j0 + p.shape[1]] = p
        return m

    mass = source_field(YEARS, NDVI_THRESHOLD)
    res["_source"] = {
        "years": list(YEARS), "ndvi_threshold": NDVI_THRESHOLD,
        "irrigated_km2": float(mass.sum()),
        "window": list(SRC),
        "note": ("unit depletion over the area above the threshold in every year "
                 "listed; the mass unit is km2 of unit depth, and only ratios of it "
                 "are reported"),
    }

    # ---------------------------------------------------------------- the footprints
    weights = {k: box_weight(lon, lat, b) for k, b in FOOTPRINTS.items()}

    # The mascon-aligned footprint: every whole mascon that carries any of the source.
    carry = sorted({int(m) for m in np.unique(lab)
                    if mass[lab == m].sum() > 0.005 * mass.sum()})
    aligned = np.isin(lab, carry)
    weights["the mascons that carry the pivots"] = np.where(aligned, area, 0.0)
    FOOT_AREA = {k: float(w.sum()) for k, w in weights.items()}
    res["_footprint_area_km2"] = FOOT_AREA
    res["_mascon_aligned_footprint"] = {
        "mascons": len(carry), "area_km2": FOOT_AREA["the mascons that carry the pivots"],
        "note": "the smallest union of whole mascons that contains the source",
    }

    # ------------------------------------------------------------------- the gains
    rows = []
    for r_km in SPREAD_KM:
        k = disc(r_km, CELL, clat)
        m_r = spread(mass, k)
        seen = mascon_field(m_r, lab, area)
        row = {"spread_km": r_km, "gain": {}}
        for name, w in weights.items():
            frac = w / np.maximum(area, 1e-12)
            true = float((m_r * frac).sum())
            got = float((seen * w).sum())
            row["gain"][name] = got / true if true > 0 else float("nan")
            row.setdefault("share_of_source_inside", {})[name] = (
                true / float(m_r.sum()) if m_r.sum() > 0 else float("nan"))
        rows.append(row)
    res["gain_by_spread"] = rows

    # The curve: concentric boxes on the source centroid, one line per source extent.
    ii, jj = np.meshgrid(clat, clon, indexing="ij")
    c_lat = float((mass * ii).sum() / mass.sum())
    c_lon = float((mass * jj).sum() / mass.sum())
    halfw = np.round(np.arange(0.25, 5.01, 0.25), 3)
    curve = {"centre": [c_lon, c_lat], "half_width_deg": halfw.tolist(), "lines": []}
    for r_km in SPREAD_KM:
        k = disc(r_km, CELL, clat)
        m_r = spread(mass, k)
        seen = mascon_field(m_r, lab, area)
        g, a_km2 = [], []
        for h in halfw:
            w = box_weight(lon, lat, (c_lon - h, c_lat - h, c_lon + h, c_lat + h))
            frac = w / np.maximum(area, 1e-12)
            true = float((m_r * frac).sum())
            g.append(float((seen * w).sum()) / true if true > 0 else float("nan"))
            a_km2.append(float(w.sum()))
        curve["lines"].append({"spread_km": r_km, "gain": g, "area_km2": a_km2})
    res["curve"] = curve

    # --------------------------------------------------------------- self-checks
    # Three limits of a mass ratio are analytic, and all three are asserted rather than
    # inspected. If any of them fails, the forward model is wrong and not the data.
    #
    #   a spatially uniform source   -> gain exactly 1 over any footprint
    #   a footprint of whole mascons -> gain exactly 1 for any source inside it
    #   spreading the source         -> conserves mass
    uni = area.copy()
    seen_u = mascon_field(uni, lab, area)
    g_uniform = {}
    for name, w in weights.items():
        frac = w / np.maximum(area, 1e-12)
        g_uniform[name] = float((seen_u * w).sum() / (uni * frac).sum())
    g_aligned = rows[0]["gain"]["the mascons that carry the pivots"]
    whole = [m for m in carry if m in inner]
    res["_checks"] = {
        "uniform_source_gain_is_one": g_uniform,
        "aligned_footprint_gain_is_one": g_aligned,
        "mass_conserved_under_spread": float(
            spread(mass, disc(100.0, CELL, clat)).sum() / mass.sum()),
        "carrying_mascons_are_whole": [len(whole), len(carry)],
    }
    for name, g in g_uniform.items():
        assert abs(g - 1.0) < 1e-9, (name, g)
    assert abs(g_aligned - 1.0) < 1e-9, g_aligned
    assert abs(res["_checks"]["mass_conserved_under_spread"] - 1.0) < 1e-3
    # A mascon clipped by the window edge would carry the wrong area and the wrong gain.
    assert len(whole) == len(carry), (whole, carry)

    # ------------------------------------------------------- what it means for L0
    pub = "the Saq box"
    g0 = {r["spread_km"]: r["gain"][pub] for r in rows}
    # An assumed gain is not a number the data can confirm or deny. What it is, once the
    # geometry is computed, is a statement about how large a footprint the account is
    # reported over, so that is the quantity it gets converted into.
    base = curve["lines"][0]
    ga, aa = np.array(base["gain"]), np.array(base["area_km2"])
    a_star = (float(np.interp(L0_PRIOR[0], ga, aa))
              if ga[0] < L0_PRIOR[0] < ga[-1] else None)
    res["_l0_prior"] = {
        "mean": L0_PRIOR[0], "sd": L0_PRIOR[1],
        "footprint": pub,
        "gain_over_that_footprint": g0[0.0],
        "footprint_area_that_reproduces_the_prior_km2": a_star,
        "aligned_footprint_area_km2": FOOT_AREA["the mascons that carry the pivots"],
        "note": ("the L0 twin assumes {:.2f}. Computed on the recovered geometry that "
                 "value belongs to a reporting footprint of about {} km2, which is the "
                 "scale of the {} mascons that carry the pivots, and not to any box "
                 "drawn around the pivot fields"
                 .format(L0_PRIOR[0],
                         "n/a" if a_star is None else "{:,.0f}".format(a_star),
                         len(carry))),
    }

    # ------------------------------------------------------------- sensitivity
    sens = []
    for years, thr in SENSITIVITY:
        m = source_field(years, thr)
        if m.sum() <= 0:
            continue
        e = {"years": list(years), "ndvi_threshold": thr,
             "persistent": len(years) > 1, "irrigated_km2": float(m.sum()), "gain": {}}
        for r_km in (0.0, 50.0):
            k = disc(r_km, CELL, clat)
            m_r = spread(m, k)
            seen = mascon_field(m_r, lab, area)
            w = weights[pub]
            frac = w / np.maximum(area, 1e-12)
            e["gain"][str(r_km)] = float((seen * w).sum()) / float((m_r * frac).sum())
        sens.append(e)
    res["sensitivity"] = sens
    bs = {str(r): g0[r] for r in (0.0, 50.0)}
    keep = [e for e in sens if e["persistent"]]
    drop = [e for e in sens if not e["persistent"]]
    pers = {k: [e["gain"][k] for e in keep] + [bs[k]] for k in bs}
    single = {k: [e["gain"][k] for e in drop] for k in bs}
    res["_sensitivity_verdict"] = {
        "persistent_delineation": {
            "gain_range": {k: [float(min(v)), float(max(v))] for k, v in pers.items()},
            "gain_sd": {k: float(np.std(v, ddof=1)) for k, v in pers.items()},
            "varied": "the NDVI threshold, over 0.35 to 0.50",
        },
        "one_year_delineation": {
            "gain_range": {k: [float(min(v)), float(max(v))] for k, v in single.items()},
            "gain_sd": {k: float(np.std(v, ddof=1)) for k, v in single.items()},
            "varied": "which single year the pivots are delineated from",
        },
        "l0_prior_sd": L0_PRIOR[1],
        "note": ("under a persistent delineation the threshold barely moves the gain, "
                 "because the geometry sets it. Delineating from one year moves it "
                 "several times as far, because a wet year enrols rainfed vegetation on "
                 "one side of a mascon boundary that runs through the pivot field. Both "
                 "spreads are reported, and both are arguments for reporting the gravity "
                 "leg over whole mascons, where the gain is one by construction and no "
                 "delineation choice can move it"),
    }
    res["_mascon_shares"] = sorted(
        ({"mascon": int(m),
          "share_of_pivot_area": float(mass[lab == m].sum() / mass.sum()),
          "area_km2": float(area[lab == m].sum())}
         for m in carry), key=lambda d: -d["share_of_pivot_area"])

    res["_mascon_uncertainty_mm"] = (float(cached["noise_mm"]) if cached is not None
                                     else mascon_noise(ee, FOOTPRINTS["the Saq box"]))
    np.savez_compressed(
        cache, stamp=stamp, lab=lab, noise_mm=res["_mascon_uncertainty_mm"],
        field_keys=np.array(list(fields), object),
        field_vals=np.array(list(fields.values()), object))
    res["_l0_grace_sigma_mm"] = 14.0

    (RES / args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.6))

    # The panel is drawn over the basin, not over the window the tessellation is
    # recovered on, which is deliberately much wider so that no mascon is clipped.
    ZOOM = (35.0, 26.5, 44.0, 33.5)
    a0 = ax[0]
    ext = [WIN[0], WIN[2], WIN[1], WIN[3]]
    a0.imshow(np.where(np.isin(lab, carry), 1.0, 0.0), origin="lower", extent=ext,
              cmap="Blues", vmin=0, vmax=3.2, interpolation="nearest")
    for m in np.unique(lab):
        sm = lab == m
        jj_ = np.where(sm.any(0))[0]
        ii_ = np.where(sm.any(1))[0]
        heavy = int(m) in carry
        a0.add_patch(plt.Rectangle(
            (lon[jj_[0]], lat[ii_[0]]), lon[jj_[-1] + 1] - lon[jj_[0]],
            lat[ii_[-1] + 1] - lat[ii_[0]], fill=False,
            ec=F.ACCENT if heavy else F.MUTED, lw=1.8 if heavy else 0.7))
    for d in res["_mascon_shares"]:
        sm = lab == d["mascon"]
        jj_ = np.where(sm.any(0))[0]
        ii_ = np.where(sm.any(1))[0]
        cx = 0.5 * (lon[jj_[0]] + lon[jj_[-1] + 1])
        cy = 0.5 * (lat[ii_[0]] + lat[ii_[-1] + 1])
        if ZOOM[0] < cx < ZOOM[2] and ZOOM[1] < cy < ZOOM[3]:
            a0.text(cx, cy, "{:.0%}\nof the\npivots".format(d["share_of_pivot_area"]),
                    ha="center", va="center", fontsize=8, color=F.ACCENT,
                    fontweight="bold")
    pm = np.ma.masked_where(mass <= 0.5, mass)
    a0.imshow(pm, origin="lower", extent=ext, cmap="autumn_r", vmin=0,
              interpolation="nearest")
    for name, b in FOOTPRINTS.items():
        a0.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                                   ec=F.WARM, lw=1.3, ls="--"))
    a0.annotate("the Saq box", (FOOTPRINTS["the Saq box"][0],
                                FOOTPRINTS["the Saq box"][3]),
                textcoords="offset points", xytext=(3, 3), fontsize=8, color=F.WARM)
    a0.set_xlim(ZOOM[0], ZOOM[2]); a0.set_ylim(ZOOM[1], ZOOM[3])
    a0.set_xlabel("longitude"); a0.set_ylabel("latitude")
    a0.set_title("The pivot field is split across {} mascons".format(len(carry)))
    a0.grid(False)

    a1 = ax[1]
    cols = [F.ACCENT, F.GREEN, F.SAND, F.WARM, F.MUTED]
    for c, line in zip(cols, curve["lines"]):
        a1.plot(np.array(line["area_km2"]) / 1e3, line["gain"], lw=1.8, color=c,
                label="{:.0f} km".format(line["spread_km"]))
    a1.axhline(L0_PRIOR[0], color=F.INK, ls=":", lw=1.2)
    a1.text(3.2, L0_PRIOR[0] + 0.03, "the gain the L0 twin assumes, {:.2f}".format(
        L0_PRIOR[0]), fontsize=8)
    marks = [("the Al Jawf pivot box", (10, 6)), ("the Saq box", (-14, 10)),
             ("the mascons that carry the pivots", (-190, 2))]
    for name, off in marks:
        a1.plot([FOOT_AREA[name] / 1e3], [rows[0]["gain"][name]], marker="o", ms=7,
                color=F.WARM, zorder=5)
        a1.annotate(name, (FOOT_AREA[name] / 1e3, rows[0]["gain"][name]),
                    textcoords="offset points", xytext=off, fontsize=8, color=F.WARM,
                    fontweight="bold")
    a1.set_xscale("log")
    a1.set_xlabel("area of the reporting footprint, thousand km²")
    a1.set_ylabel("mascon gain, mass recovered over mass present")
    a1.set_ylim(0, 1.18)
    a1.set_title("The gain is a footprint property, and it is computable")
    a1.legend(loc="lower right", fontsize=8, title="depletion spread",
              title_fontsize=8, framealpha=0.95)
    F.save(fig, FIG / args.fig)

    # ------------------------------------------------------------------- report
    print("tessellation: {} mascons in the window, {} of them whole, median area "
          "{:,.0f} km²".format(res["_tessellation"]["mascons_in_window"],
                               res["_tessellation"]["whole_mascons"],
                               res["_tessellation"]["median_area_km2"]))
    print("irrigated area {:,.0f} km² above NDVI {:.2f} in all of {}".format(
        mass.sum(), NDVI_THRESHOLD, ", ".join(str(y) for y in YEARS)))
    print("the mascon-aligned footprint is {} mascons, {:,.0f} km²".format(
        len(carry), FOOT_AREA["the mascons that carry the pivots"]))
    print()
    hdr = "  {:34s}".format("reporting footprint") + "".join(
        "{:>10s}".format("{:.0f} km".format(r)) for r in SPREAD_KM)
    print("mascon gain, by assumed depletion spread")
    print(hdr)
    for name in list(FOOTPRINTS) + ["the mascons that carry the pivots"]:
        print("  {:34s}".format(name) + "".join(
            "{:10.3f}".format(r["gain"][name]) for r in rows))
    print()
    print("the pivot field is split across {} mascons: shares {}".format(
        len(carry), ", ".join("{:.0%}".format(d["share_of_pivot_area"])
                              for d in res["_mascon_shares"])))
    sv = res["_sensitivity_verdict"]
    for k, lab in (("persistent_delineation", "threshold, persistent delineation"),
                   ("one_year_delineation", "which single year is used")):
        r = sv[k]["gain_range"]["0.0"]
        print("over {} the gain runs {:.2f} to {:.2f} varying {} (sd {:.3f})".format(
            pub, r[0], r[1], lab, sv[k]["gain_sd"]["0.0"]))
    print("the L0 twin ships a gain prior of sd {:.2f}".format(L0_PRIOR[1]))
    print("L0 assumes {:.2f}; that gain belongs to a footprint of about {} km2".format(
        L0_PRIOR[0], "n/a" if a_star is None else "{:,.0f}".format(a_star)))
    print("published mascon uncertainty over the Saq box: {} mm, against the {} mm "
          "the L0 twin assumes".format(
              "n/a" if res["_mascon_uncertainty_mm"] is None
              else "{:.1f}".format(res["_mascon_uncertainty_mm"]), 14.0))
    print("\nwrote results/{} and figures/{}".format(args.out, args.fig))


if __name__ == "__main__":
    main()
