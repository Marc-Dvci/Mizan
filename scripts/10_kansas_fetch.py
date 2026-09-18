"""L2 Kansas: retrieve the public records the rung is scored against.

Nothing here needs an account. WIMAS and WIZARD are served by the Kansas Geological
Survey for the Kansas Department of Agriculture; SSEBop is served by USGS EROS; county
precipitation comes from NOAA nClimDiv through Climate at a Glance; county boundaries
from the Census Bureau.

Usage:  python scripts/10_kansas_fetch.py [--block gmd4a] [--what wimas,wizard,ssebop]

**The transfer blocks are fetched blind.** For a block other than the published one,
the default fetch retrieves the estimator's inputs only: diversion-point locations,
water levels, the rasters, precipitation and the county polygons. The WIMAS use files,
which are the truth the block is scored against, are fetched only by an explicit
`--what wuse`, and the script refuses that until the block's predictions are recorded
in a committed `DECISION_LOG.md`, so the order "predict, run, then open the meters" is
held by the code and not by discipline.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mizan import ks_data as KD, ks_fetch as K

DATA = ROOT / "data" / "kansas"

PREDICTION_MARKER = "TRANSFER PREDICTIONS COMMITTED FOR BLOCK {block}"


def predictions_committed(block: str) -> bool:
    """True when the committed decision log carries the block's prediction marker.

    Both conditions are read from git: the marker must be in `HEAD:DECISION_LOG.md`,
    and the working copy of the log must be clean, so the marker cannot be added and
    the meters opened in the same uncommitted breath.
    """
    try:
        head = subprocess.run(["git", "show", "HEAD:DECISION_LOG.md"], cwd=ROOT,
                              capture_output=True, text=True, encoding="utf-8").stdout
        dirty = subprocess.run(["git", "status", "--porcelain", "DECISION_LOG.md"],
                               cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return False
    return PREDICTION_MARKER.format(block=block) in head and not dirty


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=str, default="gmd4a", choices=sorted(KD.BLOCKS))
    ap.add_argument("--what", type=str, default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--y0", type=int, default=2000)
    ap.add_argument("--y1", type=int, default=2025)
    args = ap.parse_args()

    blk = KD.BLOCKS[args.block]
    transfer = args.block != "gmd4a"
    suffix = "" if not transfer else "_" + args.block
    if not args.what:
        args.what = ("wizard,ssebop,mirad,hpsat,usgs,precip,wimas,wuse" if not transfer
                     else "polygons,points,wizard,ssebop,mirad,hpsat,usgs,precip")
    what = set(args.what.split(","))
    if transfer and ("wuse" in what or "wimas" in what):
        if not predictions_committed(args.block):
            sys.exit("refusing to fetch the use files for block {!r}: the marker\n  {}\n"
                     "is not in the committed DECISION_LOG.md, or the log has uncommitted "
                     "changes.\nRecord the predictions, commit, then fetch."
                     .format(args.block, PREDICTION_MARKER.format(block=args.block)))
        print("predictions for block {} are committed; opening the use files"
              .format(args.block))

    if "polygons" in what:
        print("Census county boundaries, block " + args.block)
        K.fetch_polygons(DATA, blk["fips"], suffix)

    if "points" in what:
        print("WIMAS irrigation diversion points, locations only, block " + args.block)
        for code in blk["counties"]:
            t = time.time()
            K.fetch_county_points(code, DATA)
            print("  {} ({}) {:.0f}s".format(blk["names"][code], code, time.time() - t))

    if "wizard" in what:
        print("WIZARD water levels, block " + args.block)
        K.fetch_wizard(DATA, counties=blk["counties"], workers=args.workers,
                       fips=blk["fips"], names=blk["names"], suffix=suffix)

    if "ssebop" in what:
        print("SSEBop annual actual evapotranspiration, CONUS 1 km")
        for y in range(args.y0, args.y1 + 1):
            t = time.time()
            p = K.ssebop_year(y, DATA / "ssebop")
            print("  {} {:.1f} MB {:.0f}s".format(
                y, p.stat().st_size / 1e6, time.time() - t))

    if "mirad" in what:
        print("MIrAD-US irrigated agriculture, 250 m")
        K.fetch_mirad(DATA)

    if "hpsat" in what:
        print("USGS High Plains saturated thickness, 2009, 500 m")
        K.fetch_hpsat(DATA)

    if "usgs" in what:
        print("USGS High Plains specific-yield and hydraulic-conductivity maps, 1998")
        K.fetch_usgs_fields(DATA)

    if "precip" in what:
        print("NOAA nClimDiv annual county precipitation, the recharge forcing")
        K.fetch_precipitation(DATA, y0=args.y0, y1=min(args.y1, 2024),
                              fips=blk["fips"], suffix=suffix)

    if "wuse" in what:
        print("WIMAS water-use files with the measurement code on every report")
        for code in blk["counties"]:
            t = time.time()
            K.fetch_county_use(code, DATA)
            print("  {} ({}) {:.0f}s".format(blk["names"][code], code, time.time() - t))

    if "wimas" in what:
        print("WIMAS reported annual pumping by water right, from the history page")
        for code in blk["counties"]:
            t = time.time()
            K.fetch_county(code, DATA, workers=args.workers)
            print("  {} ({}) {:.0f}s".format(blk["names"][code], code, time.time() - t))


if __name__ == "__main__":
    main()
