"""L2 Kansas: calibrate the interannual amplitude without an oracle.

The closure's county-year anomaly correlates with the metered anomaly at well below one.
Under a mean-absolute-error metric a correlation below one makes the optimal amplitude
smaller than the estimate's own, so an estimate can carry real year-to-year information
and still be scored at or below a flat estimate until its amplitude is corrected.

The amplitude is corrected by one scalar. Fitting that scalar against the meters is an
oracle and cannot be reported. Fitting it leave-one-county-out is not: each county's
anomaly is shrunk by a factor estimated from the other five counties only, which is the
operational statement that a few metered counties calibrate the amplitude for the rest.
That is the requirement the L0 value-of-information layer already reached from the other
direction.

Reported for every row, plus the oracle factor, so the distance between the two is
visible rather than assumed.

    python scripts/15_kansas_shrink.py [--tag _v3]
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
ROWS = ("ETH", "ET", "H")
LABEL = {"ETH": "evapotranspiration + heads, closure",
         "ET": "evapotranspiration only",
         "H": "heads only",
         "BASELINE": "open loop at 0.80"}
GRID = np.linspace(0.0, 1.5, 151)


def anomaly(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=1, keepdims=True)


def best_factor(ah: np.ndarray, at: np.ndarray, rows: np.ndarray) -> float:
    """The shrink factor minimising anomaly MAE over the counties in `rows`."""
    err = np.array([np.abs(s * ah[rows] - at[rows]).mean() for s in GRID])
    return float(GRID[int(err.argmin())])


def loco(ah: np.ndarray, at: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-county shrink, each fitted on the other counties only."""
    nc = ah.shape[0]
    out = np.empty(nc)
    for i in range(nc):
        rest = np.array([j for j in range(nc) if j != i])
        out[i] = best_factor(ah, at, rest)
    return out, ah * out[:, None]


def score(hat: np.ndarray, q_true: np.ndarray, signal: float) -> dict:
    ah, at = anomaly(hat), anomaly(q_true)
    return {"mae_mcm": float(np.abs(hat - q_true).mean() / 1e6),
            "anomaly_mae_mcm": float(np.abs(ah - at).mean() / 1e6),
            "anomaly_skill": 1.0 - float(np.abs(ah - at).mean()) / (signal * 1e6)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    ref = np.load(RES / f"kansas_posterior_ETH{args.tag}.npz")
    q_true = ref["q_true"]
    at = anomaly(q_true)
    signal = float(np.abs(at).mean() / 1e6)
    out: dict = {"_signal_mcm": signal, "_tag": args.tag}

    cand = {"BASELINE": ref["et_obs"] / 0.80}
    for k in ROWS:
        p = RES / f"kansas_posterior_{k}{args.tag}.npz"
        if p.exists():
            cand[k] = np.load(p)["ens"].mean(axis=0)

    print(f"county-year anomaly signal {signal:.2f} Mm3/yr mean absolute, "
          f"{q_true.shape[0]} counties x {q_true.shape[1]} years\n")
    print(f"{'estimate':40s} {'r':>6s} {'raw':>7s} {'LOCO':>7s} {'oracle':>7s} "
          f"{'factors':>28s}")

    for k, hat in cand.items():
        ah = anomaly(hat)
        r = float(np.corrcoef(ah.ravel(), at.ravel())[0, 1])
        s_or = best_factor(ah, at, np.arange(ah.shape[0]))
        fac, ah_lo = loco(ah, at)
        rec = {"label": LABEL.get(k, k), "r": r,
               "raw": score(hat, q_true, signal),
               "loco": score(hat.mean(axis=1, keepdims=True) + ah_lo, q_true, signal),
               "oracle": score(hat.mean(axis=1, keepdims=True) + s_or * ah,
                               q_true, signal),
               "oracle_factor": s_or,
               "loco_factors": fac.tolist(),
               "dispersion": float(np.abs(ah).mean() / np.abs(at).mean())}
        out[k] = rec
        print(f"{rec['label']:40s} {r:6.2f} {rec['raw']['anomaly_skill']:7.2f} "
              f"{rec['loco']['anomaly_skill']:7.2f} "
              f"{rec['oracle']['anomaly_skill']:7.2f}   "
              + " ".join(f"{v:.2f}" for v in fac))

    (RES / f"kansas_shrink{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote results/kansas_shrink{args.tag}.json")
    print("Skill is against a flat-in-time estimate. The LOCO column is the one that can "
          "be reported: every county's factor comes from the other five.")


if __name__ == "__main__":
    main()
