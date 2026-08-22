"""Numerical guards for the allocation claims exposed in the submission."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mizan import allocation as AL, config as C


def member(pcs: float = -1.0) -> AL.MemberSurrogate:
    return AL.MemberSurrogate(
        h_ref=np.zeros((1, AL.HORIZON_Y)),
        R=np.zeros((1, C.NDIST, AL.HORIZON_Y)),
        pcs=np.array([pcs]),
        ssv_b=1.0,
        area=np.array([1.0]),
        loss_hist=0.0,
    )


def test_empirical_cvar_fractionally_weights_the_quantile_boundary():
    values = np.arange(1.0, 11.0)
    assert AL.empirical_cvar(values, 0.85) == pytest.approx((10.0 + 0.5 * 9.0) / 1.5)
    with pytest.raises(ValueError):
        AL.empirical_cvar(values, 1.0)


def test_policy_postcheck_covers_volume_bounds_floor_and_chance_constraint():
    q = np.full((C.NDIST, AL.HORIZON_Y), 1.0e6)
    total = float(q.sum())
    cap = np.full_like(q, 2.0e6)
    floor = np.full(C.NDIST, 0.5 * q.sum(axis=1)[0])
    ok = AL.validate_policy([member()], q, q, total, cap, floor, chance=0.95)
    assert ok["feasible"]

    bad_cap = cap.copy()
    bad_cap[0, 0] = 0.5e6
    rejected = AL.validate_policy([member()], q, q, total, bad_cap, floor, chance=0.95)
    assert not rejected["feasible"]
    assert rejected["cap_violation_m3"] == pytest.approx(0.5e6)


def test_infeasible_chance_problem_cannot_return_a_publishable_policy():
    q_ref = np.full((C.NDIST, AL.HORIZON_Y), 1.0e6)
    total = float(q_ref.sum())
    cap = np.full_like(q_ref, 2.0e6)
    floor = np.zeros(C.NDIST)
    result = AL.optimise([member(pcs=1.0)], q_ref, total, cap, floor,
                         chance=0.95, solvers=("CLARABEL", "SCS"))
    assert not result.feasible
    assert result.policy is None

