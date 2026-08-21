"""L2 Kansas: retrieve the public records the rung is scored against.

Nothing here needs an account. WIMAS and WIZARD are served by the Kansas Geological
Survey for the Kansas Department of Agriculture; SSEBop is served by USGS EROS; county
precipitation comes from NOAA nClimDiv through Climate at a Glance.

Usage:  python scripts/10_kansas_fetch.py [--what wimas,wizard,ssebop] [--workers 4]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mizan import ks_fetch as K

DATA = ROOT / "data" / "kansas"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", type=str,
                    default="wizard,ssebop,mirad,hpsat,precip,wimas")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--y0", type=int, default=2000)
    ap.add_argument("--y1", type=int, default=2025)
    args = ap.parse_args()
    what = set(args.what.split(","))

    if "wizard" in what:
        print("WIZARD water levels, six counties of Northwest Kansas")
        K.fetch_wizard(DATA, workers=args.workers)

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

    if "precip" in what:
        print("NOAA nClimDiv annual county precipitation, the recharge forcing")
        K.fetch_precipitation(DATA, y0=args.y0, y1=min(args.y1, 2024))

    if "wimas" in what:
        print("WIMAS metered annual pumping, GMD 4 counties")
        for code, name in K.GMD4.items():
            t = time.time()
            K.fetch_county(code, DATA, workers=args.workers)
            print("  {} ({}) {:.0f}s".format(name, code, time.time() - t))


if __name__ == "__main__":
    main()
