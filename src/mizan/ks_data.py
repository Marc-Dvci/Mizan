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

COUNTIES = ["CN", "RA", "DC", "SH", "TH", "SD"]
COUNTY_NAME = {"CN": "Cheyenne", "RA": "Rawlins", "DC": "Decatur",
               "SH": "Sherman", "TH": "Thomas", "SD": "Sheridan"}

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
    tiers = [["CN", "RA", "DC"], ["SH", "TH", "SD"]]
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


def build_region(points: dict, delr_km: float = DELR_KM) -> Region:
    box = county_boxes(points)
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
    rec = json.loads((DATA / "wizard_levels.json").read_text())
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
    cache = DATA / "ssebop_region.npz"
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
    cache = DATA / "mirad_region.npz"
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
    cache = DATA / "hpsat_region.npz"
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


def precipitation() -> np.ndarray:
    """Annual county precipitation, mm/yr, shape (6, nyear), from NOAA nClimDiv."""
    d = json.loads((DATA / "precip_annual.json").read_text())
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
