"""Assemble the L2 Kansas rung from the retrieved public records.

The study region is the six-county block of Northwest Kansas over the Ogallala:
Cheyenne, Rawlins and Decatur on the northern tier, Sherman, Thomas and Sheridan on the
southern one. Kansas counties in this part of the state are rectangles of the Public
Land Survey System, so the block is a clean three by two arrangement and each county is
a management district in the same sense as the L0 experiment.

Three products come out of here:

* `Region`, the model grid and the county map on it;
* the observations the estimator is allowed to see, which are satellite actual
  evapotranspiration and annual water levels;
* the withheld truth, which is per-county metered annual pumping from WIMAS.

Nothing in the observation set is derived from the water-use reports.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "kansas"

# The county blocks the rung can be assembled on. `gmd4a` is the published block, and
# it keeps the tier geometry and the file names the published `_v3` results were built
# from, so that block reproduces them byte for byte. The other three are the transfer
# blocks: chosen as contiguous groups of six counties before any of their inputs was
# fetched, scored blind, and never tuned on. Their county map comes from the Census
# county polygons rather than from tiers, because Finney and Ford are not rectangles.
#
# Every code is the two-letter county code WIMAS itself uses; `fips` is the county's
# FIPS code, which is what WIZARD, nClimDiv and the Census key the county by.
BLOCKS = {
    "gmd4a": dict(
        label="Northwest Kansas, GMD4 north (the published block)",
        counties=["CN", "RA", "DC", "SH", "TH", "SD"],
        names={"CN": "Cheyenne", "RA": "Rawlins", "DC": "Decatur",
               "SH": "Sherman", "TH": "Thomas", "SD": "Sheridan"},
        fips={"CN": "023", "RA": "153", "DC": "039",
              "SH": "181", "TH": "193", "SD": "179"},
        tiers=[["CN", "RA", "DC"], ["SH", "TH", "SD"]]),
    "west": dict(
        label="West Kansas, GMD4 south and GMD1",
        counties=["WA", "LG", "GO", "GL", "WH", "SC"],
        names={"WA": "Wallace", "LG": "Logan", "GO": "Gove",
               "GL": "Greeley", "WH": "Wichita", "SC": "Scott"},
        fips={"WA": "199", "LG": "109", "GO": "063",
              "GL": "071", "WH": "203", "SC": "171"},
        tiers=None),
    "gmd3w": dict(
        label="Southwest Kansas, GMD3 west",
        counties=["HM", "KE", "ST", "GT", "MT", "SV"],
        names={"HM": "Hamilton", "KE": "Kearny", "ST": "Stanton",
               "GT": "Grant", "MT": "Morton", "SV": "Stevens"},
        fips={"HM": "075", "KE": "093", "ST": "187",
              "GT": "067", "MT": "129", "SV": "189"},
        tiers=None),
    "gmd3e": dict(
        label="Southwest Kansas, GMD3 east",
        counties=["FI", "HS", "GY", "SW", "ME", "FO"],
        names={"FI": "Finney", "HS": "Haskell", "GY": "Gray",
               "SW": "Seward", "ME": "Meade", "FO": "Ford"},
        fips={"FI": "055", "HS": "081", "GY": "069",
              "SW": "175", "ME": "119", "FO": "057"},
        tiers=None),
}
TRANSFER_BLOCKS = ("west", "gmd3w", "gmd3e")

BLOCK = "gmd4a"
COUNTIES = list(BLOCKS[BLOCK]["counties"])
COUNTY_NAME = dict(BLOCKS[BLOCK]["names"])


def set_block(name: str) -> None:
    """Point the module at one county block.

    The county list is mutated in place rather than rebound, so every module that
    imported it keeps seeing the current block; `ks_run.set_block` recomputes the
    parameter layout that depends on its length.
    """
    global BLOCK
    if name not in BLOCKS:
        raise ValueError("unknown block {!r}; one of {}".format(name, sorted(BLOCKS)))
    BLOCK = name
    COUNTIES[:] = BLOCKS[name]["counties"]
    COUNTY_NAME.clear()
    COUNTY_NAME.update(BLOCKS[name]["names"])


def _sfx() -> str:
    """File-name suffix of the current block; empty for the published one."""
    return "" if BLOCK == "gmd4a" else "_" + BLOCK


DELR_KM = 2.0
YEAR0, YEAR1 = 2000, 2024
AF_TO_M3 = 1233.4818
FT_TO_M = 0.3048


# --------------------------------------------------------------------------- geometry
@dataclass(frozen=True)
class Region:
    """A local equal-spacing grid over the six-county block.

    Distances are metres in a local equirectangular frame centred on the block. Over
    150 km at this latitude the frame is accurate to better than the 2 km cell.
    """

    lon0: float
    lat0: float
    nrow: int
    ncol: int
    delr_m: float
    county: np.ndarray          # (nrow, ncol) index into COUNTIES, -1 outside
    # "tiers": the published rectangle, whose lateral boundary is the grid edge.
    # "polygons": a Census-polygon block, whose lateral boundary is wherever an active
    # cell meets an inactive one, because the block need not fill its bounding box.
    layout: str = "tiers"

    @property
    def kx(self) -> float:
        return 111_320.0 * np.cos(np.deg2rad(self.lat0))

    @property
    def ky(self) -> float:
        return 110_574.0

    def to_xy(self, lon, lat):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky

    def to_lonlat(self, x, y):
        return self.lon0 + np.asarray(x) / self.kx, self.lat0 + np.asarray(y) / self.ky

    def cell_of(self, lon, lat):
        """Row and column of the cell containing each point, -1 where outside."""
        x, y = self.to_xy(lon, lat)
        col = np.floor(x / self.delr_m).astype(int)
        row = np.floor(y / self.delr_m).astype(int)
        bad = (col < 0) | (col >= self.ncol) | (row < 0) | (row >= self.nrow)
        col = np.where(bad, -1, col)
        row = np.where(bad, -1, row)
        return row, col

    def centers_lonlat(self):
        cx = (np.arange(self.ncol) + 0.5) * self.delr_m
        cy = (np.arange(self.nrow) + 0.5) * self.delr_m
        X, Y = np.meshgrid(cx, cy)
        return self.to_lonlat(X, Y)

    @property
    def area_m2(self) -> float:
        return self.delr_m ** 2


def county_boxes(points: dict) -> dict:
    """Longitude and latitude edges of each county, from its points of diversion.

    Counties here are Public Land Survey System rectangles. Taking the extreme
    coordinates of the licensed diversion points recovers the rectangle to within the
    margin where no well happens to sit, which is smaller than one model cell.
    """
    box = {}
    for c in COUNTIES:
        lon = np.array([p["lon"] for p in points[c]])
        lat = np.array([p["lat"] for p in points[c]])
        box[c] = dict(w=lon.min(), e=lon.max(), s=lat.min(), n=lat.max())

    # Share the edges between neighbours so the block tiles without gaps.
    tiers = BLOCKS[BLOCK]["tiers"]
    for tier in tiers:
        for a, b in zip(tier, tier[1:]):
            m = 0.5 * (box[a]["e"] + box[b]["w"])
            box[a]["e"] = box[b]["w"] = m
    for a, b in zip(tiers[0], tiers[1]):
        m = 0.5 * (box[a]["s"] + box[b]["n"])
        box[a]["s"] = box[b]["n"] = m
    w = min(box[c]["w"] for c in COUNTIES)
    e = max(box[c]["e"] for c in COUNTIES)
    s = min(box[c]["s"] for c in COUNTIES)
    n = max(box[c]["n"] for c in COUNTIES)
    for c in COUNTIES:
        box[c]["w"] = w if box[c]["w"] - w < 0.12 else box[c]["w"]
        box[c]["e"] = e if e - box[c]["e"] < 0.12 else box[c]["e"]
        box[c]["s"] = s if box[c]["s"] - s < 0.12 else box[c]["s"]
        box[c]["n"] = n if n - box[c]["n"] < 0.12 else box[c]["n"]
    return box


def load_polygons() -> dict:
    """The Census county polygons of the current block, lon/lat rings per county."""
    return json.loads((DATA / f"county_polygons{_sfx()}.json").read_text())


def build_region_polygons(delr_km: float = DELR_KM) -> Region:
    """The grid and county map of a block from the Census county polygons.

    The grid is the bounding box of the block's polygons; a cell belongs to the county
    whose polygon contains its centre and is inactive where none does. The same local
    equirectangular frame and cell size as the published block.
    """
    from matplotlib.path import Path as MPath

    polys = load_polygons()
    rings = {c: [np.asarray(r, dtype=float) for r in polys[c]] for c in COUNTIES}
    allpts = np.concatenate([r for c in COUNTIES for r in rings[c]])
    lon0, lat0 = allpts[:, 0].min(), allpts[:, 1].min()
    lon1, lat1 = allpts[:, 0].max(), allpts[:, 1].max()
    kx = 111_320.0 * np.cos(np.deg2rad(lat0))
    ncol = int(np.ceil((lon1 - lon0) * kx / (delr_km * 1000.0)))
    nrow = int(np.ceil((lat1 - lat0) * 110_574.0 / (delr_km * 1000.0)))
    r = Region(lon0, lat0, nrow, ncol, delr_km * 1000.0, np.full((nrow, ncol), -1),
               layout="polygons")
    LON, LAT = r.centers_lonlat()
    pts = np.column_stack([LON.ravel(), LAT.ravel()])
    cty = np.full(nrow * ncol, -1)
    for i, c in enumerate(COUNTIES):
        inside = np.zeros(nrow * ncol, dtype=bool)
        for ring in rings[c]:
            inside |= MPath(ring).contains_points(pts)
        cty[inside & (cty < 0)] = i
    return Region(lon0, lat0, nrow, ncol, delr_km * 1000.0, cty.reshape(nrow, ncol),
                  layout="polygons")


def build_region(points: dict = None, delr_km: float = DELR_KM) -> Region:
    if BLOCKS[BLOCK]["tiers"] is None:
        return build_region_polygons(delr_km)
    box = county_boxes(points if points is not None else load_points())
    lon0 = min(box[c]["w"] for c in COUNTIES)
    lat0 = min(box[c]["s"] for c in COUNTIES)
    lon1 = max(box[c]["e"] for c in COUNTIES)
    lat1 = max(box[c]["n"] for c in COUNTIES)

    kx = 111_320.0 * np.cos(np.deg2rad(lat0))
    ncol = int(np.ceil((lon1 - lon0) * kx / (delr_km * 1000.0)))
    nrow = int(np.ceil((lat1 - lat0) * 110_574.0 / (delr_km * 1000.0)))

    r = Region(lon0, lat0, nrow, ncol, delr_km * 1000.0, np.full((nrow, ncol), -1))
    LON, LAT = r.centers_lonlat()
    cty = np.full((nrow, ncol), -1)
    for i, c in enumerate(COUNTIES):
        b = box[c]
        sel = (LON >= b["w"]) & (LON < b["e"]) & (LAT >= b["s"]) & (LAT < b["n"])
        cty[sel] = i
    return Region(lon0, lat0, nrow, ncol, delr_km * 1000.0, cty)


# --------------------------------------------------------------------------- WIMAS
def load_points() -> dict:
    return {c: json.loads((DATA / f"wimas_{c}.json").read_text())["points"]
            for c in COUNTIES}


def points_only() -> bool:
    """True when the block's point records carry no use table (a blind block)."""
    return any(json.loads((DATA / f"wimas_{c}.json").read_text()).get("points_only")
               for c in COUNTIES)


def metered_annual() -> tuple[np.ndarray, dict]:
    """County-annual metered irrigation pumping, m3/yr, shape (6, nyear).

    Use is filed per water right. A right whose diversion points fall in more than one
    county has its reported volume split between them in proportion to the number of
    points, which is the only split the public record supports.
    """
    years = np.arange(YEAR0, YEAR1 + 1)
    per_right: dict[str, dict] = {}
    right_counties: dict[str, list[str]] = {}
    n_missing = 0
    for c in COUNTIES:
        rec = json.loads((DATA / f"wimas_{c}.json").read_text())
        for p in rec["points"]:
            right_counties.setdefault(p["wr"], []).append(c)
        for wr, series in rec["use"].items():
            if "_error" in series:
                n_missing += 1
                continue
            merged = per_right.setdefault(wr, {})
            for key, yearly in series.items():
                for y, v in yearly.items():
                    merged[int(y)] = max(merged.get(int(y), 0.0), float(v))

    q = np.zeros((len(COUNTIES), years.size))
    for wr, series in per_right.items():
        cs = right_counties.get(wr, [])
        if not cs:
            continue
        share = {}
        for c in cs:
            share[c] = share.get(c, 0) + 1
        tot = sum(share.values())
        for y, af in series.items():
            if not (YEAR0 <= y <= YEAR1):
                continue
            for c, k in share.items():
                q[COUNTIES.index(c), y - YEAR0] += af * AF_TO_M3 * k / tot
    return q, {"n_rights": len(per_right), "n_missing": n_missing,
               "years": years.tolist()}


# The per-report measurement code WIMAS carries on every water-use record, from the
# code table in KGS Open-File Report 2005-30. Three codes mean a meter was read: A (or
# 7), metered acre-feet; M (or 8), metered gallons; I, metered acre-inches. G is a
# quantity computed from hours of pump operation and a pump rate; F is a field
# inspection. Every other code is a non-use or administrative report and carries no
# volume. The lower-case forms occur three times in the six counties.
METER_CODES = frozenset({"A", "7", "M", "8", "I", "a", "m"})
HOURS_CODES = frozenset({"G"})
FIELD_CODES = frozenset({"F"})

# The first year every report in the block is meter-coded, to within 1.2 per cent of
# the volume. GMD4 records 2009 as the first year all of its wells were metered.
METERED_ERA_YEAR0 = 2009


def _use_records(county: str) -> list[dict]:
    """One row per (water right, point of diversion, year) from the WIMAS use file.

    The file joins one row per aquifer code, so a report can appear twice with
    everything else equal; five pairs in the six counties pair a filed report with a
    blank one. Each (right, point, year) keeps the row carrying the largest volume.
    """
    import csv
    rows = list(csv.reader((DATA / f"wimas_wuse_{county}.txt").open(
        newline="", encoding="latin-1")))
    head = [x.strip() for x in rows[0]]
    best: dict[tuple, dict] = {}
    for r in rows[1:]:
        if len(r) != len(head):
            continue
        d = dict(zip(head, [x.strip() for x in r]))
        try:
            d["af"] = float(d["af_used"])
        except ValueError:
            d["af"] = 0.0
        k = (d["wr_id"], d["pdiv_id"], d["wua_year"])
        if k not in best or d["af"] > best[k]["af"]:
            best[k] = d
    return list(best.values())


def reported_annual() -> tuple[np.ndarray, np.ndarray, dict]:
    """County-annual reported irrigation pumping, m3/yr, split by how it was measured.

    Returns `(q, share, meta)`: `q` has shape (3, 6, nyear) with the volume that is
    meter-coded, hours-times-rate coded, and field-inspection coded on the first axis;
    `share[c, y]` is the metered fraction of the county-year total. Every point of
    diversion files its own report, and the row carries the point's county, so a right
    whose points fall in two counties is attributed by where each report was filed.

    This is the whole record. `metered_annual()` above read one point of diversion per
    right from the history page and so under-counts rights with several reporting
    points; it is kept because the published `_v3` scores were computed against it.
    """
    years = np.arange(YEAR0, YEAR1 + 1)
    q = np.zeros((3, len(COUNTIES), years.size))
    n = np.zeros((3, len(COUNTIES), years.size), dtype=int)
    other = 0.0
    for ci, c in enumerate(COUNTIES):
        for d in _use_records(c):
            y = int(d["wua_year"])
            if not (YEAR0 <= y <= YEAR1) or d["af"] <= 0.0:
                continue
            code = d["wur_code"]
            k = (0 if code in METER_CODES else 1 if code in HOURS_CODES
                 else 2 if code in FIELD_CODES else -1)
            if k < 0:
                other += d["af"]
                continue
            cj = COUNTIES.index(d["county"]) if d["county"] in COUNTIES else ci
            q[k, cj, y - YEAR0] += d["af"] * AF_TO_M3
            n[k, cj, y - YEAR0] += 1
    tot = q.sum(axis=0)
    share = np.where(tot > 0, q[0] / np.where(tot > 0, tot, 1.0), np.nan)
    return q, share, {"years": years.tolist(), "n_reports": n.tolist(),
                      "volume_on_other_codes_af": other,
                      "metered_era_year0": METERED_ERA_YEAR0}


def diversion_weights(region: Region) -> np.ndarray:
    """Per-cell share of each county's pumping, from the licensed diversion points.

    Locations are licence data. The volumes are not: they are the withheld truth.
    """
    pts = load_points()
    w = np.zeros((len(COUNTIES), region.nrow, region.ncol))
    for i, c in enumerate(COUNTIES):
        lon = np.array([p["lon"] for p in pts[c]])
        lat = np.array([p["lat"] for p in pts[c]])
        row, col = region.cell_of(lon, lat)
        ok = (row >= 0) & (col >= 0)
        np.add.at(w[i], (row[ok], col[ok]), 1.0)
        w[i][region.county != i] = 0.0
        if w[i].sum() > 0:
            w[i] /= w[i].sum()
    return w


# --------------------------------------------------------------------------- WIZARD
def water_levels(region: Region) -> dict:
    """Annual water-table elevation at every well with a usable record.

    Kansas measures its network in winter, when the aquifer has recovered from the
    season. Only measurements from December to March are kept, and a year is dated by
    the January it belongs to, so one value per well per year enters.
    """
    rec = json.loads((DATA / f"wizard_levels{_sfx()}.json").read_text())
    years = np.arange(YEAR0, YEAR1 + 1)
    wells = []
    for w in rec["wells"]:
        if "_error" in w or not w.get("levels") or w.get("altitude_ft") in (None, 0):
            continue
        head = {}
        for date, dtw in w["levels"].items():
            y, m, _ = (int(v) for v in date.split("-"))
            if m in (12,):
                y += 1
            elif m not in (1, 2, 3):
                continue
            if YEAR0 <= y <= YEAR1:
                head.setdefault(y, []).append(
                    (w["altitude_ft"] - dtw) * FT_TO_M)
        if len(head) < 8:
            continue
        row, col = region.cell_of(w["lon"], w["lat"])
        if row < 0 or col < 0 or region.county[row, col] < 0:
            continue
        series = np.full(years.size, np.nan)
        for y, vals in head.items():
            series[y - YEAR0] = float(np.mean(vals))
        wells.append(dict(usgs_id=w["usgs_id"], lon=w["lon"], lat=w["lat"],
                          row=int(row), col=int(col),
                          altitude_m=w["altitude_ft"] * FT_TO_M,
                          depth_m=(w.get("depth_ft") or 0.0) * FT_TO_M,
                          head=series))
    return {"years": years, "wells": wells}


# --------------------------------------------------------------------------- SSEBop
def _read_window(tif: Path, region: Region) -> np.ndarray:
    """SSEBop annual actual evapotranspiration, mm/yr, on the model grid."""
    import rasterio
    from rasterio.windows import from_bounds

    LON, LAT = region.centers_lonlat()
    with rasterio.open(tif) as src:
        pad = 0.05
        win = from_bounds(LON.min() - pad, LAT.min() - pad,
                          LON.max() + pad, LAT.max() + pad, src.transform)
        arr = src.read(1, window=win).astype(float)
        tr = src.window_transform(win)
    inv = ~tr
    cols, rows = inv * (LON, LAT)
    cols = np.clip(np.round(cols - 0.5).astype(int), 0, arr.shape[1] - 1)
    rows = np.clip(np.round(rows - 0.5).astype(int), 0, arr.shape[0] - 1)
    out = arr[rows, cols]
    out[out > 6000] = np.nan
    return out


def _cache_key(region: Region) -> np.ndarray:
    """Identity of the grid a cached raster was sampled on.

    Shape alone is not the identity: two regions can share it and differ in origin, and
    a cache hit on the wrong origin would move the whole basin with nothing raised.
    """
    return np.array([region.lon0, region.lat0, region.nrow, region.ncol,
                     region.delr_m], dtype=float)


def _load_cache(path: Path, region: Region, shape) -> np.ndarray | None:
    if not path.exists():
        return None
    z = np.load(path)
    if "key" not in z or not np.allclose(z["key"], _cache_key(region)):
        return None
    return z["arr"] if z["arr"].shape == shape else None


def _save_cache(path: Path, region: Region, arr: np.ndarray) -> None:
    np.savez_compressed(path, arr=arr, key=_cache_key(region))


def evapotranspiration(region: Region) -> np.ndarray:
    """Annual actual evapotranspiration on the model grid, mm/yr, shape (nyear, r, c)."""
    years = np.arange(YEAR0, YEAR1 + 1)
    cache = DATA / f"ssebop_region{_sfx()}.npz"
    hit = _load_cache(cache, region, (years.size, region.nrow, region.ncol))
    if hit is not None:
        return hit
    arr = np.stack([_read_window(DATA / "ssebop" / f"ssebop_{y}.tif", region)
                    for y in years])
    _save_cache(cache, region, arr)
    return arr


MIRAD_EPOCHS = {2002: "02", 2007: "07", 2012: "12", 2017: "17"}


def _mirad_epoch(tag: str, region: Region, sub: int = 8) -> np.ndarray:
    """Irrigated fraction of every model cell, from the 250 m MIrAD-US map."""
    import glob
    import rasterio
    from rasterio.warp import transform as rio_transform

    path = glob.glob(str(DATA / "mirad" / "**" / f"mirad250_{tag}v4.tif"),
                     recursive=True)[0]
    with rasterio.open(path) as src:
        arr = src.read(1)
        off = (np.arange(sub) + 0.5) / sub - 0.5
        xs, ys = [], []
        for dy in off:
            for dx in off:
                lon = region.lon0 + ((np.arange(region.ncol) + 0.5 + dx)
                                     * region.delr_m) / region.kx
                lat = region.lat0 + ((np.arange(region.nrow) + 0.5 + dy)
                                     * region.delr_m) / region.ky
                L, A = np.meshgrid(lon, lat)
                xs.append(L.ravel())
                ys.append(A.ravel())
        X = np.concatenate(xs)
        Y = np.concatenate(ys)
        px, py = rio_transform("EPSG:4326", src.crs, X.tolist(), Y.tolist())
        r, c = rasterio.transform.rowcol(src.transform, px, py)
        r = np.clip(np.asarray(r), 0, src.height - 1)
        c = np.clip(np.asarray(c), 0, src.width - 1)
        v = arr[r, c].reshape(sub * sub, region.nrow, region.ncol)
    return (v > 0).mean(axis=0)


def irrigated_fraction(region: Region) -> np.ndarray:
    """Irrigated fraction per cell and year, shape (nyear, nrow, ncol).

    MIrAD-US publishes 2002, 2007, 2012 and 2017. Irrigated extent changes slowly, so
    the intervening years are interpolated linearly and the ends are held.
    """
    cache = DATA / f"mirad_region{_sfx()}.npz"
    years = np.arange(YEAR0, YEAR1 + 1)
    hit = _load_cache(cache, region, (years.size, region.nrow, region.ncol))
    if hit is not None:
        return hit
    ep = sorted(MIRAD_EPOCHS)
    maps = np.stack([_mirad_epoch(MIRAD_EPOCHS[y], region) for y in ep])
    out = np.empty((years.size, region.nrow, region.ncol))
    for i, y in enumerate(years):
        yc = min(max(y, ep[0]), ep[-1])
        k = int(np.searchsorted(ep, yc, side="right") - 1)
        k = min(k, len(ep) - 2)
        w = (yc - ep[k]) / (ep[k + 1] - ep[k])
        out[i] = (1 - w) * maps[k] + w * maps[k + 1]
    _save_cache(cache, region, out)
    return out


def irrigation_et(et: np.ndarray, frac: np.ndarray, region: Region,
                  pool: bool = True) -> tuple:
    """Irrigation consumptive use per county-year, m3/yr, and its standard error.

    A 1 km evapotranspiration pixel over a quarter-section pivot landscape is a mixture
    of irrigated and dryland ground, so no threshold recovers the irrigated signal from
    the pixel values alone. Within one county and one year the pixels instead satisfy

        ET(cell) = ET_dry + (ET_irr - ET_dry) * f(cell) + noise,

    with `f` the irrigated fraction of the cell from the 250 m irrigation map. The
    ordinary least squares slope of that line is the irrigation excess in mm, and
    multiplying it by the county's irrigated area gives the volume. The mixture is
    resolved rather than thresholded away, and the estimate does not depend on the
    resolution of the evapotranspiration product.

    The standard error of the slope is returned with it, because a county with almost no
    irrigation has almost no leverage and its estimate has to enter the likelihood at
    the weight it deserves.

    **The slope has to be pooled across the block.** A county-by-county fit is an
    extrapolation from the irrigated fractions that county happens to contain out to
    f = 1. Where the map tops out near 0.2 that is a five-fold extrapolation and it
    returns a fifth of the volume. The endmembers are climatic and are shared across a
    164 by 96 km block, so the slope is estimated once per year over every cell with a
    separate intercept per county, which is the within estimator, and each county's own
    slope is shrunk toward it by its own leverage:

        b_i <- b_block + tau^2 / (tau^2 + se_i^2) * (b_i - b_block),

    with `tau` the between-county spread of the slope in excess of what the regression
    standard errors already explain. A county with tight leverage keeps its own value; a
    county with none is carried by the block. No metered volume enters any of this.
    Pass `pool=False` for the county-by-county fit that this replaces.
    """
    ny = et.shape[0]
    nc = len(COUNTIES)
    slope = np.zeros((nc, ny))
    slope_se = np.zeros((nc, ny))
    area = np.zeros((nc, ny))
    ok = np.zeros((nc, ny), dtype=bool)
    b_blk = np.zeros(ny)
    se_blk = np.zeros(ny)

    for t in range(ny):
        xs, ys = [], []
        for i in range(nc):
            sel = region.county == i
            x = frac[t][sel]
            y = et[t][sel]
            good = np.isfinite(y) & np.isfinite(x)
            x, y = x[good], y[good]
            area[i, t] = np.nansum(frac[t][sel]) * region.area_m2
            n = x.size
            if n < 20:
                continue
            xc, yc = x - x.mean(), y - y.mean()
            sxx = (xc ** 2).sum()
            if sxx <= 0:
                continue
            b = (xc * yc).sum() / sxx
            s2 = ((yc - b * xc) ** 2).sum() / max(n - 2, 1)
            slope[i, t] = b
            slope_se[i, t] = np.sqrt(s2 / sxx)
            ok[i, t] = True
            xs.append(xc)
            ys.append(yc)
        if not xs:
            continue
        xc = np.concatenate(xs)
        yc = np.concatenate(ys)
        den = (xc ** 2).sum()
        b_blk[t] = (xc * yc).sum() / den
        s2 = ((yc - b_blk[t] * xc) ** 2).sum() / max(xc.size - len(xs) - 1, 1)
        se_blk[t] = np.sqrt(s2 / den)

    if pool and ok.any():
        d2 = ((slope - b_blk[None, :])[ok] ** 2).mean()
        tau2 = max(0.0, float(d2 - (slope_se[ok] ** 2).mean()))
        w = tau2 / (tau2 + slope_se ** 2) if tau2 > 0 else np.zeros_like(slope_se)
        slope = np.where(ok, b_blk[None, :] + w * (slope - b_blk[None, :]), 0.0)
        slope_se = np.where(
            ok, np.sqrt(w * slope_se ** 2 + (se_blk ** 2)[None, :]), 0.0)

    vol = np.where(ok, slope * 1e-3 * area, 0.0)
    se = np.where(ok, slope_se * 1e-3 * area, 0.0)
    return vol, se


def irrigated_area(frac: np.ndarray, region: Region) -> np.ndarray:
    """Irrigated area per county-year, m2, shape (6, nyear)."""
    out = np.zeros((len(COUNTIES), frac.shape[0]))
    for i in range(len(COUNTIES)):
        sel = region.county == i
        out[i] = frac[:, sel].sum(axis=1) * region.area_m2
    return out


def saturated_thickness(region: Region) -> np.ndarray:
    """Published saturated thickness of the High Plains aquifer on the grid, metres.

    USGS `hp_satthk09`, the 2009 saturated-thickness grid of the High Plains aquifer,
    500 m, EPSG:5070, published in feet. It is an independent observation of the
    aquifer's geometry: no water-use report enters it, and nothing in the inversion is
    scored against it, so it can be used to set the base of the layer rather than left
    to be estimated.

    The grid reports zero outside the mapped aquifer and where the aquifer has been
    dewatered. A zero cell is filled with the median of the mapped cells in its own
    county, and a county with none is filled with the block median, because a zero is
    "not mapped here" and not "no aquifer here" as far as a 2 km cell is concerned.
    """
    cache = DATA / f"hpsat_region{_sfx()}.npz"
    hit = _load_cache(cache, region, (region.nrow, region.ncol))
    if hit is not None:
        return hit

    import rasterio
    from rasterio.warp import transform as _warp

    src_path = DATA / "hpsat" / "hp_satthk09"
    LON, LAT = region.centers_lonlat()
    with rasterio.open(src_path) as src:
        xs, ys = _warp("EPSG:4326", src.crs, LON.ravel().tolist(), LAT.ravel().tolist())
        arr = src.read(1).astype(float)
        nod = src.nodata
        inv = ~src.transform
    if nod is not None:
        arr[arr == nod] = np.nan
    arr[arr < -1e30] = np.nan
    cols, rows = inv * (np.asarray(xs), np.asarray(ys))
    r = np.clip(np.round(rows - 0.5).astype(int), 0, arr.shape[0] - 1)
    c = np.clip(np.round(cols - 0.5).astype(int), 0, arr.shape[1] - 1)
    b = (arr[r, c].reshape(LON.shape)) * 0.3048

    # Below a metre the cell is either outside the mapped aquifer or dewatered on the
    # published surface, and neither is a thickness a 2 km cell can carry.
    good = np.isfinite(b) & (b > 1.0)
    fill = float(np.median(b[good])) if good.any() else 20.0
    for i in range(len(COUNTIES)):
        sel = region.county == i
        g = sel & good
        v = float(np.median(b[g])) if g.any() else fill
        b[sel & ~good] = v
    b[~np.isfinite(b) | (b <= 1.0)] = fill
    _save_cache(cache, region, b)
    return b


# ------------------------------------------------------------ USGS property maps
def _e00_polygons(path: Path, table: str):
    """Polygons and their (MAJOR1, MINOR1) class bounds from an ArcInfo E00 export.

    ARC carries the arcs, PAL the arc list of every polygon, and the polygon attribute
    table the class each polygon belongs to. Polygon 1 is the universe and is skipped.
    """
    import re
    lines = path.read_text(encoding="latin-1").splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("ARC ")) + 1
    arcs = {}
    while True:
        h = lines[i].split()
        if len(h) >= 7 and h[0] == "-1":
            break
        aid, npt = int(h[1]), int(h[6])
        i += 1
        pts = []
        while len(pts) < npt:
            nums = re.findall(r"-?\d\.\d+E[+-]\d\d", lines[i])
            pts += [(float(nums[j]), float(nums[j + 1])) for j in range(0, len(nums), 2)]
            i += 1
        arcs[aid] = pts
    i = next(k for k, l in enumerate(lines) if l.startswith("PAL ")) + 1
    pals = []
    while True:
        n = int(lines[i][:10])
        if n == -1:
            break
        i += 1
        trip = []
        while len(trip) < n:
            nums = [int(x) for x in lines[i].split()]
            trip += [tuple(nums[j:j + 3]) for j in range(0, len(nums), 3)]
            i += 1
        pals.append(trip)
    i = next(k for k, l in enumerate(lines) if l.startswith(table + ".PAT "))
    nrec = int(lines[i].split()[-1])
    i += 1
    while not re.match(r"^\s*-?\d\.\d+E", lines[i]):
        i += 1
    attrs = [(int(lines[i + r][-12:-6]), int(lines[i + r][-6:])) for r in range(nrec)]
    out = []
    for pid, pal in enumerate(pals, start=1):
        if pid == 1:
            continue
        rings, cur = [], []
        for a, _, _ in pal:
            if a == 0:
                if cur:
                    rings.append(cur)
                cur = []
                continue
            pts = arcs[abs(a)]
            cur += pts if a > 0 else pts[::-1]
        if cur:
            rings.append(cur)
        rings = [r for r in rings if len(r) >= 4]
        if rings:
            out.append((rings, attrs[pid - 1]))
    return out


def usgs_field(region: Region, which: str) -> np.ndarray:
    """A published High Plains property map on the model grid.

    `which` is "sy", the specific yield of Cederstrand and Becker (1998, OFR 98-414),
    returned as a fraction, or "k", their hydraulic conductivity (OFR 98-548), returned
    in m/d. Both are class maps; a cell takes the midpoint of its class. They are
    observations of the aquifer's material into which no water-use report enters, so the
    inversion can put its storage coefficient and its conductivity prior on them rather
    than estimate a single block-wide value of each.
    """
    import pyproj
    from rasterio import features
    from rasterio.transform import from_origin

    cache = DATA / f"usgs_{which}_region{_sfx()}.npz"
    hit = _load_cache(cache, region, (region.nrow, region.ncol))
    if hit is not None:
        return hit
    src = {"sy": ("ofr98-414.e00", "SY", 0.01), "k": ("ofr98-548.e00", "COND", FT_TO_M)}
    fname, table, scale = src[which]
    polys = _e00_polygons(DATA / "usgs_fields" / fname, table)
    albers = pyproj.CRS.from_proj4("+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 "
                                   "+lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m")
    tr = pyproj.Transformer.from_crs("EPSG:4269", albers, always_xy=True)
    LON, LAT = region.centers_lonlat()
    X, Y = tr.transform(LON, LAT)
    res = 500.0
    x0, y1 = X.min() - 5e3, Y.max() + 5e3
    W = int((X.max() + 5e3 - x0) / res) + 1
    H = int((y1 - Y.min() + 5e3) / res) + 1
    shapes = [({"type": "Polygon",
                "coordinates": [[list(q) for q in r] + [list(r[0])] for r in rings]},
               0.5 * (maj + mino) * scale) for rings, (maj, mino) in polys]
    grid = features.rasterize(shapes, out_shape=(H, W),
                              transform=from_origin(x0, y1, res, res),
                              fill=np.nan, dtype="float64")
    out = grid[((y1 - Y) / res).astype(int), ((X - x0) / res).astype(int)]
    # A class of zero is the map's "not mapped"; fill it and the unmapped cells from
    # the county's own median, as the saturated-thickness field is filled.
    out[out <= 0] = np.nan
    for i in range(len(COUNTIES)):
        m = region.county == i
        if np.isnan(out[m]).all():
            continue
        out[m & np.isnan(out)] = np.nanmedian(out[m])
    out[np.isnan(out)] = np.nanmedian(out)
    _save_cache(cache, region, out)
    return out


def precipitation() -> np.ndarray:
    """Annual county precipitation, mm/yr, shape (6, nyear), from NOAA nClimDiv."""
    d = json.loads((DATA / f"precip_annual{_sfx()}.json").read_text())
    years = np.arange(YEAR0, YEAR1 + 1)
    return np.array([[d[c][str(y)] for y in years] for c in COUNTIES])


def recharge_weight() -> np.ndarray:
    """Per-county, per-year multiplier on the mean recharge, shape (6, nyear).

    Recharge is the residual of a large precipitation against a large evaporative
    demand, so it cannot be held constant across a record whose precipitation varies by
    a factor of 2.3. What is estimated stays one mean rate; its time structure is taken
    from the observed precipitation and is not free.

    The multiplier is each county's precipitation divided by that county's own mean over
    the record, so the record mean of the multiplier is one by construction and the
    estimated mean recharge keeps its published prior. No water-use record enters it.
    """
    p = precipitation()
    return p / p.mean(axis=1, keepdims=True)
