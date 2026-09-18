"""L3: is the deformation leg observable over the target basin's own pivots?

The entry's closure has four legs and the Kansas rung exercises two of them. The
deformation leg has never been observed by this pipeline on a real aquifer, and the
question a juror asks about the Saq is not whether interferometry works in general but
whether it works **over irrigated ground in Al Jawf**, where active centre pivots
decorrelate the radar and the signal has to be read on the desert between them.

That is a measurement, not an argument, and it needs no account. COMET's LiCSAR service
publishes processed Sentinel-1 interferograms for frame `116A_05991_141313`, whose
footprint contains the Wadi As-Sirhan pivot field. This script reads them over the same
area-of-interest box the Al Jawf rung uses and reports three things:

* **coherence over the pivots against coherence over the desert**, with the pivot mask
  taken from the ESA WorldCover 10 m cropland class rather than from the radar, so the
  classification is not circular;
* **how much of the basin carries a usable phase**, at two coherence thresholds;
* **the line-of-sight rate** the published epochs support, by least squares over the
  interferogram network, referenced to unirrigated desert, with the standard error the
  record's own span allows.

What this cannot do is stated by the record rather than by opinion: the frame publishes
{N} epochs over one season, so the rate it supports is a seasonal one and the published
Saq subsidence rates of 4 to 15 mm/yr are compared against what that span can resolve.

    python scripts/32_saq_insar.py [--max-pairs N]

Writes results/saq_insar.json and figures/fig19_saq_insar.png.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RES = ROOT / "results"
FIG = ROOT / "figures"
CACHE = ROOT / "data" / "saq_insar"

UA = {"User-Agent": "Mozilla/5.0"}
FRAME = "116A_05991_141313"
TRACK = "116"
LICS = ("https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products/"
        f"{TRACK}/{FRAME}/")
WORLDCOVER = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
              "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
CROPLAND = 40                      # ESA WorldCover class code

# The same box the Al Jawf rung delineates its pivots in, read from that rung's own
# output so the two cannot drift apart.
AOI = json.loads((RES / "aljawf.json").read_text())["_aoi"]

# Sentinel-1 C band. One radian of unwrapped phase is lambda / (4 pi) of range change.
LAMBDA_MM = 55.465
RAD_TO_MM = LAMBDA_MM / (4.0 * np.pi)


def listing(url: str) -> list[str]:
    h = urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                               timeout=60).read().decode()
    return [x for x in re.findall(r'href="([^"?]+)"', h) if not x.startswith("/")]


def pair_files(pair: str) -> dict:
    """The real product URLs behind a pair's index stub."""
    h = urllib.request.urlopen(
        urllib.request.Request(LICS + "interferograms/" + pair, headers=UA),
        timeout=60).read().decode()
    out = {}
    for u in re.findall(r"href='([^']+)'", h):
        name = u.rsplit("/", 1)[1]
        if name.endswith(".geo.unw.tif"):
            out["unw"] = u
        elif name.endswith(".geo.cc.tif"):
            out["cc"] = u
    return out


def read_window(url: str, bounds, cache: Path, band: int = 1):
    """One AOI window of a remote GeoTIFF, cached locally as a .npy."""
    if cache.exists():
        z = np.load(cache)
        return z["arr"], tuple(z["tr"])
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open("/vsicurl/" + url) as s:
        w = from_bounds(*bounds, s.transform)
        arr = s.read(band, window=w).astype(np.float32)
        tr = s.window_transform(w)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, arr=arr, tr=np.array(tr[:6]))
    return arr, tuple(tr[:6])


def cropland_mask(bounds, shape, tr) -> np.ndarray:
    """Share of each radar pixel that WorldCover calls cropland.

    The 10 m map is read over the same box and averaged onto the 100 m radar grid, so a
    pixel's value is the irrigated fraction of it rather than a threshold.
    """
    cache = CACHE / "worldcover_aoi.npz"
    if cache.exists():
        z = np.load(cache)
        if z["arr"].shape == shape:
            return z["arr"]
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    import rasterio
    from rasterio.windows import from_bounds
    lon0, lat0, lon1, lat1 = bounds
    acc = np.zeros(shape, dtype=np.float32)
    n = np.zeros(shape, dtype=np.float32)
    tiles = sorted({"N{:02d}E{:03d}".format(int(np.floor(la / 3) * 3),
                                            int(np.floor(lo / 3) * 3))
                    for lo in (lon0, lon1) for la in (lat0, lat1)})
    a, b, c, d, e, f = tr
    for tile in tiles:
        try:
            with rasterio.open("/vsicurl/" + WORLDCOVER.format(tile=tile)) as s:
                bb = s.bounds
                box = (max(lon0, bb.left), max(lat0, bb.bottom),
                       min(lon1, bb.right), min(lat1, bb.top))
                if box[0] >= box[2] or box[1] >= box[3]:
                    continue
                w = from_bounds(*box, s.transform)
                arr = s.read(1, window=w)
                twr = s.window_transform(w)
        except Exception as exc:                       # network, not logic
            print("  worldcover {}: {}".format(tile, repr(exc)[:70]))
            continue
        rows, cols = np.nonzero(np.ones(arr.shape, dtype=bool))
        lon = twr.c + (cols + 0.5) * twr.a
        lat = twr.f + (rows + 0.5) * twr.e
        row = ((lat - f) / e).astype(int)
        col = ((lon - c) / a).astype(int)
        ok = (row >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])
        np.add.at(acc, (row[ok], col[ok]),
                  (arr.ravel()[ok] == CROPLAND).astype(np.float32))
        np.add.at(n, (row[ok], col[ok]), 1.0)
    frac = np.where(n > 0, acc / np.maximum(n, 1.0), np.nan)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, arr=frac)
    return frac


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=40)
    args = ap.parse_args()

    lon0, lat0, lon1, lat1 = AOI
    bounds = (lon0, lat0, lon1, lat1)

    pairs = sorted(listing(LICS + "interferograms/"))
    pairs = [p for p in pairs if re.fullmatch(r"\d{8}_\d{8}", p)][: args.max_pairs]
    print(f"frame {FRAME}: {len(pairs)} interferograms published")

    cc_stack, unw_stack, used = [], [], []
    tr = None
    for p in pairs:
        f = pair_files(p)
        if "unw" not in f or "cc" not in f:
            continue
        try:
            cc, tr = read_window(f["cc"], bounds, CACHE / f"{p}_cc.npz")
            unw, tr = read_window(f["unw"], bounds, CACHE / f"{p}_unw.npz")
        except Exception as exc:
            print(f"  {p}: {repr(exc)[:80]}")
            continue
        # The two products round their windows independently, so a pair can come back
        # one row or column apart. Everything is cropped to the common shape.
        h = min(cc.shape[0], unw.shape[0])
        w = min(cc.shape[1], unw.shape[1])
        if not np.isfinite(np.nanmean(cc)) or np.nanmean(cc) / 255.0 < 0.05:
            # A pair whose window carries no usable phase: LiCSAR publishes it but the
            # box is empty or decorrelated end to end. Dropped rather than averaged in.
            print(f"  {p}: no usable phase over the box, dropped")
            continue
        cc_stack.append(cc[:h, :w] / 255.0)
        unw_stack.append(unw[:h, :w])
        used.append(p)
        print(f"  {p}: {cc.shape}, mean coherence {np.nanmean(cc) / 255.0:.2f}")
    if not used:
        sys.exit("no interferograms could be read")

    epochs = sorted({e for p in used for e in p.split("_")})
    print(f"{len(used)} usable interferograms over {len(epochs)} epochs, "
          f"{epochs[0]} to {epochs[-1]}")
    h = min(x.shape[0] for x in cc_stack + unw_stack)
    w = min(x.shape[1] for x in cc_stack + unw_stack)
    C = np.stack([x[:h, :w] for x in cc_stack])
    U = np.stack([x[:h, :w] for x in unw_stack])
    shape = C.shape[1:]
    frac = cropland_mask(bounds, shape, tr)

    # An unwrapped value of exactly zero is LiCSAR's no-data, not zero displacement.
    U = np.where(U == 0.0, np.nan, U)
    valid = np.isfinite(frac)
    pivot = valid & (frac > 0.5)
    desert = valid & (frac < 0.02)
    # The reference and the control have to be different ground, or the control is the
    # reference and returns zero by construction. Distance from the nearest cropland
    # pixel splits the desert: the far field is the reference every interferogram is
    # levelled against, the near field is an independent control that has to come back
    # at zero if the levelling is doing its job.
    try:
        from scipy import ndimage
        dist_px = ndimage.distance_transform_edt(~(valid & (frac > 0.2)))
    except ImportError:
        dist_px = np.full(shape, 1e6)
    km_px = abs(tr[0]) * 111.0
    far = desert & (dist_px * km_px > 20.0)
    near = desert & (dist_px * km_px > 3.0) & (dist_px * km_px <= 12.0)

    cm = np.nanmean(C, axis=0)
    out = {"_meta": {
        "frame": FRAME, "track": TRACK, "aoi": AOI, "n_pairs": len(used),
        "epochs": epochs, "first_epoch": epochs[0], "last_epoch": epochs[-1],
        "pixel_deg": abs(tr[0]), "shape": list(shape),
        "pivot_mask": ("ESA WorldCover 2021 class 40, cropland, averaged onto the radar "
                       "grid; a pixel is pivot above 0.5 and desert below 0.02"),
        "n_pivot_px": int(pivot.sum()), "n_desert_px": int(desert.sum()),
        "n_reference_px": int((far & (cm > 0.3)).sum()),
        "n_control_px": int(near.sum()),
        "pixel_m": float(abs(tr[0]) * 111000.0),
        "radians_to_mm": RAD_TO_MM}}

    # ------------------------------------------------------------ 1. coherence
    out["coherence"] = {
        "mean_over_pivots": float(np.nanmean(cm[pivot])),
        "mean_over_desert": float(np.nanmean(cm[desert])),
        "share_of_pivots_above_0.3": float(np.nanmean(cm[pivot] > 0.3)),
        "share_of_pivots_above_0.5": float(np.nanmean(cm[pivot] > 0.5)),
        "share_of_desert_above_0.3": float(np.nanmean(cm[desert] > 0.3)),
        "share_of_desert_above_0.5": float(np.nanmean(cm[desert] > 0.5)),
        "by_pair_pivot": {p: float(np.nanmean(C[i][pivot])) for i, p in enumerate(used)},
        "by_pair_desert": {p: float(np.nanmean(C[i][desert])) for i, p in enumerate(used)},
        "reading": ("the leg is read on the ground between the fields; what matters is "
                    "whether that ground holds phase at a 12-day repeat")}

    # ------------------------------------------------------------ 2. the rate
    # Least squares for per-epoch displacement from the interferogram network, then a
    # linear rate through the epochs. Every interferogram is referenced to the median of
    # the coherent desert first, which removes the orbital and atmospheric level each
    # one carries.
    t = np.array([(np.datetime64(f"{e[:4]}-{e[4:6]}-{e[6:]}")
                   - np.datetime64(f"{epochs[0][:4]}-{epochs[0][4:6]}-{epochs[0][6:]}"))
                  / np.timedelta64(1, "D") for e in epochs], dtype=float)
    ref = far & (cm > 0.3)
    if ref.sum() < 500:                       # the far field is small on some frames
        ref = desert & (cm > 0.3)
    G = np.zeros((len(used), len(epochs)))
    for i, p in enumerate(used):
        a, b = p.split("_")
        G[i, epochs.index(b)] = 1.0
        G[i, epochs.index(a)] = -1.0
    G = G[:, 1:]                                        # first epoch is the datum

    def rate_over(mask: np.ndarray) -> tuple[float, float, int]:
        """Least-squares line-of-sight rate over a mask, mm/yr, and its standard error."""
        sel = mask & (cm > 0.3)
        if sel.sum() < 50:
            return float("nan"), float("nan"), int(sel.sum())
        d = np.array([np.nanmedian(U[i][sel]) - np.nanmedian(U[i][ref])
                      for i in range(len(used))]) * RAD_TO_MM
        keep = np.isfinite(d)
        x, res, *_ = np.linalg.lstsq(G[keep], d[keep], rcond=None)
        disp = np.concatenate([[0.0], x])
        A = np.column_stack([np.ones_like(t), t])
        beta, *_ = np.linalg.lstsq(A, disp, rcond=None)
        fit = A @ beta
        dof = max(len(t) - 2, 1)
        s2 = float(((disp - fit) ** 2).sum() / dof)
        cov = s2 * np.linalg.inv(A.T @ A)
        return float(beta[1] * 365.25), float(np.sqrt(cov[1, 1]) * 365.25), int(sel.sum())

    r_p, se_p, n_p = rate_over(pivot)
    r_d, se_d, n_d = rate_over(near)
    span_yr = float((t[-1] - t[0]) / 365.25)
    out["rate_los_mm_yr"] = {
        "pivots": {"rate": r_p, "se": se_p, "n_px": n_p},
        "desert_control": {"rate": r_d, "se": se_d, "n_px": n_d,
                           "what": ("unirrigated ground 3 to 12 km from the nearest "
                                    "cropland, levelled against the far field beyond "
                                    "20 km, so it is an independent control and not "
                                    "the reference itself")},
        "span_years": span_yr,
        "detectable_at_2se_mm_yr": 2.0 * se_p,
        "published_saq_subsidence_mm_yr": [4.0, 15.0],
        "reading": ("the published frame spans one season, so this is a seasonal rate "
                    "with the error that span supports; it is reported against what it "
                    "can resolve rather than as an annual velocity")}

    RES.mkdir(exist_ok=True)
    (RES / "saq_insar.json").write_text(json.dumps(out, indent=2))

    # ------------------------------------------------------------------ the figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
        ext = [lon0, lon1, lat0, lat1]
        ax[0].imshow(frac, extent=ext, origin="upper", cmap="YlGn", vmin=0, vmax=1)
        ax[0].set_title("Irrigated fraction, ESA WorldCover cropland", fontsize=10)
        ax[0].set_xlabel("longitude")
        ax[0].set_ylabel("latitude")

        im = ax[1].imshow(cm, extent=ext, origin="upper", cmap="magma", vmin=0, vmax=1)
        ax[1].set_title("Mean Sentinel-1 coherence, {} interferograms".format(len(used)),
                        fontsize=10)
        ax[1].set_xlabel("longitude")
        fig.colorbar(im, ax=ax[1], fraction=0.046)

        bins = np.linspace(0, 1, 41)
        ax[2].hist(cm[desert][np.isfinite(cm[desert])], bins=bins, density=True,
                   color="#d9a441", alpha=0.75, label="desert")
        ax[2].hist(cm[pivot][np.isfinite(cm[pivot])], bins=bins, density=True,
                   color="#2b6ca3", alpha=0.75, label="pivot fields")
        ax[2].axvline(0.3, color="k", lw=1, ls="--")
        ax[2].set_xlabel("mean coherence")
        ax[2].set_ylabel("density")
        ax[2].set_title("The leg is readable between the fields", fontsize=10)
        ax[2].legend(fontsize=9)
        fig.tight_layout()
        FIG.mkdir(exist_ok=True)
        fig.savefig(FIG / "fig19_saq_insar.png", dpi=160)
    except ImportError:
        pass

    # ------------------------------------------------------------------ the report
    c = out["coherence"]
    print(f"\npixels: {out['_meta']['n_pivot_px']:,} pivot, "
          f"{out['_meta']['n_desert_px']:,} desert, at "
          f"{out['_meta']['pixel_m']:.0f} m")
    print(f"mean coherence: pivots {c['mean_over_pivots']:.2f}, "
          f"desert {c['mean_over_desert']:.2f}")
    print(f"share above 0.3: pivots {c['share_of_pivots_above_0.3']:.2f}, "
          f"desert {c['share_of_desert_above_0.3']:.2f}")
    print(f"share above 0.5: pivots {c['share_of_pivots_above_0.5']:.2f}, "
          f"desert {c['share_of_desert_above_0.5']:.2f}")
    r = out["rate_los_mm_yr"]
    print(f"\nline-of-sight rate over {r['span_years']:.2f} years of published epochs:")
    print(f"  pivots  {r['pivots']['rate']:+7.1f} +- {r['pivots']['se']:.1f} mm/yr "
          f"on {r['pivots']['n_px']:,} coherent pixels")
    print(f"  desert  {r['desert_control']['rate']:+7.1f} +- "
          f"{r['desert_control']['se']:.1f} mm/yr on {r['desert_control']['n_px']:,}")
    print(f"  what this span resolves at two standard errors: "
          f"{r['detectable_at_2se_mm_yr']:.1f} mm/yr, against a published Saq range of "
          f"{r['published_saq_subsidence_mm_yr'][0]:.0f} to "
          f"{r['published_saq_subsidence_mm_yr'][1]:.0f}")
    print("\nwrote results/saq_insar.json")


if __name__ == "__main__":
    main()
