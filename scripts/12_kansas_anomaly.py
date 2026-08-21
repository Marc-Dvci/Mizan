"""L2 Kansas: separate what is easy from what is hard.

Irrigated area is a good predictor of how much a county pumps, so a prior built from an
irrigation map and one published applied depth already lands close on the county mean.
That makes the total, and to a large extent the county level, the easy part. What no such
prior can see is the year-to-year variation.

This script scores the same estimates twice: on the level, and on the county-year anomaly
about each county's own mean over the record. A flat-in-time estimate scores exactly the
size of the anomaly signal on the second metric, so anything below it is variation the
estimator actually recovered.

    python scripts/12_kansas_anomaly.py [--tag _v3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mizan import ks_data as K

RES = ROOT / "results"
ROWS = ("ETH", "H", "ET")
LABEL = {"ETH": "evapotranspiration + heads, closure",
         "H": "heads only", "ET": "evapotranspiration only"}


def anomaly(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    tag = args.tag

    ks = json.loads((RES / f"kansas{tag}.json").read_text())
    out = {}

    ref = np.load(RES / f"kansas_posterior_ETH{tag}.npz")
    q_true = ref["q_true"]
    et = ref["et_obs"]
    signal = float(np.abs(anomaly(q_true)).mean() / 1e6)

    def score(hat, name):
        out[name] = {
            "mae_mcm": float(np.abs(hat - q_true).mean() / 1e6),
            "anomaly_mae_mcm": float(
                np.abs(anomaly(hat) - anomaly(q_true)).mean() / 1e6),
        }
        out[name]["anomaly_skill"] = 1.0 - out[name]["anomaly_mae_mcm"] / signal

    score(np.tile(q_true.mean(axis=1, keepdims=True), (1, q_true.shape[1])),
          "FLAT")
    score(et / 0.80, "BASELINE")
    e_star = float(ks["BASELINE_ORACLE"]["label"].rsplit(" ", 1)[-1])
    score(et / e_star, "BASELINE_ORACLE")
    for k in ROWS:
        p = RES / f"kansas_posterior_{k}{tag}.npz"
        if p.exists():
            score(np.load(p)["ens"].mean(axis=0), k)

    out["_signal_mcm"] = signal
    (RES / f"kansas_anomaly{tag}.json").write_text(json.dumps(out, indent=2))

    print(f"county-year anomaly signal: {signal:.2f} Mm3/yr mean absolute\n")
    print(f"{'estimate':44s} {'MAE':>8s} {'anomaly MAE':>12s} {'skill':>7s}")
    names = {"FLAT": "each county's own mean, flat in time",
             "BASELINE": "open loop at 0.80",
             "BASELINE_ORACLE": f"open loop, constant fitted at {e_star:.3f}", **LABEL}
    for k, v in out.items():
        if k.startswith("_"):
            continue
        print(f"{names.get(k, k):44s} {v['mae_mcm']:8.2f} "
              f"{v['anomaly_mae_mcm']:12.2f} {v['anomaly_skill']:7.2f}")


if __name__ == "__main__":
    main()
