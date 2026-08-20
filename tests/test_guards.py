"""Guards on the parts of the pipeline whose failure would leave no visible trace.

Each guard is paired with a corruption that must make it fail. A check that has never
failed has not been tested, so every assertion here that something is right is
followed by an assertion that the same check rejects a deliberately broken input.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mizan import config as C, forcing as F, fields, estimator as E
from mizan import inversion as I, model as M, observations as O

TRUTH_WS = ROOT / "runs" / "truth"


def listed_storage_release(lst: Path) -> float:
    """Net water released from storage over the run, m3, from MODFLOW's own budget."""
    txt = lst.read_text(encoding="utf-8", errors="ignore")
    blk = txt[txt.rfind("CUMULATIVE VOLUME"):]
    cum = {}
    for line in blk.splitlines():
        m = re.match(r"\s+([A-Z0-9\-]+)\s*=\s*([0-9.E+\-]+)", line)
        if m:
            cum.setdefault(m.group(1), []).append(float(m.group(2)))
    # each key appears twice in the cumulative column, once under IN and once under OUT
    total = 0.0
    for k in ("STO-SS", "STO-SY", "CSUB-CGELASTIC", "CSUB-ELASTIC", "CSUB-INELASTIC"):
        v = cum.get(k, [0.0, 0.0])
        total += v[0] - v[1]
    return total


@pytest.mark.skipif(not (TRUTH_WS / "mizan.lst").exists(), reason="truth run absent")
def test_storage_identity_matches_modflow_budget():
    """The GRACE operator integrates compaction and specific yield. That integral has
    to be the same number MODFLOW writes in its own volumetric budget."""
    st = O.read_state(TRUTH_WS, C.TRUTH)
    mine = st.storage_depletion[-1]
    theirs = listed_storage_release(TRUTH_WS / "mizan.lst")
    assert abs(mine - theirs) / theirs < 0.01

    # the same check must reject a state whose specific-yield term is dropped
    broken = O.State(head0=st.head0, head=st.head, comp=st.comp,
                     area_m2=st.area_m2, sy=0.0)
    assert abs(broken.storage_depletion[-1] - theirs) / theirs > 0.5


@pytest.mark.skipif(not (TRUTH_WS / "mizan.cmpi").exists(), reason="truth run absent")
def test_permanent_loss_matches_the_inelastic_budget_term():
    """The storage capacity reported as permanently destroyed must be the inelastic
    term MODFLOW itself accounted for."""
    inel = O.read_inelastic(TRUTH_WS, C.TRUTH)
    mine = inel[-1].sum() * C.TRUTH.delr_m ** 2
    txt = (TRUTH_WS / "mizan.lst").read_text(encoding="utf-8", errors="ignore")
    blk = txt[txt.rfind("CUMULATIVE VOLUME"):]
    v = [float(m.group(1)) for m in
         re.finditer(r"CSUB-INELASTIC\s*=\s*([0-9.E+\-]+)", blk)]
    theirs = v[0] - v[2]
    assert abs(mine - theirs) / theirs < 0.005

    # the total compaction field is not the same quantity and must not pass this check
    st = O.read_state(TRUTH_WS, C.TRUTH)
    total = st.comp[-1].sum() * C.TRUTH.delr_m ** 2
    assert abs(total - theirs) / theirs > 0.05


def test_esmda_subspace_equals_dense():
    rng = np.random.default_rng(0)
    npar, nobs, ne = 5, 20, 40
    A = rng.standard_normal((nobs, npar))
    sd = np.full(nobs, 0.3)
    d_obs = A @ rng.standard_normal(npar) + sd * rng.standard_normal(nobs)
    X = rng.standard_normal((npar, ne)) + 2.0
    D = A @ X
    alpha = 2.0

    dX = X - X.mean(1, keepdims=True)
    dD = D - D.mean(1, keepdims=True)
    Cxd = dX @ dD.T / (ne - 1)
    Cdd = dD @ dD.T / (ne - 1)
    pert = d_obs[:, None] + np.sqrt(alpha) * sd[:, None] * np.random.default_rng(7).standard_normal((nobs, ne))
    dense = X + Cxd @ np.linalg.solve(Cdd + alpha * np.diag(sd ** 2), pert - D)

    got = I.esmda_update(X, D, d_obs, sd, alpha, None, np.random.default_rng(7), rtps=0.0)
    assert np.abs(dense - got).max() < 1e-8


def test_esmda_reduces_misfit_on_a_linear_problem():
    rng = np.random.default_rng(3)
    npar, nobs, ne = 8, 60, 80
    A = rng.standard_normal((nobs, npar))
    sd = np.full(nobs, 0.2)
    xt = rng.standard_normal(npar)
    d_obs = A @ xt + sd * rng.standard_normal(nobs)
    X = rng.standard_normal((npar, ne)) + 3.0
    phi0 = (((A @ X - d_obs[:, None]) / sd[:, None]) ** 2).mean()
    for a in I.alpha_schedule(4):
        X = I.esmda_update(X, A @ X, d_obs, sd, float(a), None, rng)
    phi1 = (((A @ X - d_obs[:, None]) / sd[:, None]) ** 2).mean()
    assert phi1 < phi0 / 10.0


def test_rtps_restores_spread_without_moving_the_mean():
    """Relaxation to prior spread must widen a collapsed ensemble and leave its mean
    where the analysis put it."""
    rng = np.random.default_rng(11)
    npar, nobs, ne = 6, 400, 60
    A = rng.standard_normal((nobs, npar))
    sd = np.full(nobs, 0.05)
    d_obs = A @ rng.standard_normal(npar) + sd * rng.standard_normal(nobs)
    X = rng.standard_normal((npar, ne)) + 2.0
    plain = I.esmda_update(X, A @ X, d_obs, sd, 1.0, None, np.random.default_rng(4), rtps=0.0)
    relax = I.esmda_update(X, A @ X, d_obs, sd, 1.0, None, np.random.default_rng(4), rtps=0.7)
    assert relax.std(axis=1).mean() > plain.std(axis=1).mean() * 1.2
    assert np.abs(relax.mean(axis=1) - plain.mean(axis=1)).max() < 1e-8
    assert plain.std(axis=1).mean() < X.std(axis=1).mean()


def test_et_operator_is_blind_to_pre_canopy_irrigation():
    """The evapotranspiration leg must not see water applied before the canopy exists.
    That blindness is the reason an open-loop account is a lower bound."""
    q = np.full((C.NDIST, C.NYEAR), 1.0e8)
    pre = np.full(C.NDIST, 0.12)
    eta = np.full(C.NDIST, 0.80)
    et = O.op_et(q, eta, pre).reshape(C.NDIST, C.NYEAR, 12)
    for m in C.PREPLANT_MONTHS:
        assert np.allclose(et[:, :, m], 0.0)
    seen = et.sum(axis=2) / (eta[:, None] * q)
    assert np.allclose(seen, 1.0 - pre[0])

    # with no pre-canopy share the same operator must see everything
    full = O.op_et(q, eta, np.zeros(C.NDIST)).reshape(C.NDIST, C.NYEAR, 12).sum(axis=2)
    assert np.allclose(full / (eta[:, None] * q), 1.0)


def test_localisation_is_causal_and_leg_aware():
    """No observation may inform pumping that had not happened when it was taken, and
    an evapotranspiration retrieval may not inform the conductivity field."""
    tr = np.load(ROOT / "results" / "truth.npz")
    geom = O.Geometry(well_xy=tr["well_xy"], well_seen=tr["well_seen"],
                      insar_xy=tr["insar_xy"], insar_ref_xy=tr["insar_ref_xy"],
                      insar_epochs=np.arange(C.INSAR_STACK_MONTHS - 1, C.NPER,
                                             C.INSAR_STACK_MONTHS))
    rho = I.taper(geom, C.EST)
    meta = I.obs_meta(geom)
    yobs = meta["t"] // 12

    row = E.LAYOUT["logq"].start + 3 * C.NYEAR + 10          # district 3, year 10
    assert rho[row][yobs < 10].max() == 0.0
    assert rho[row][(yobs >= 10) & (meta["leg"] == 1)].min() == 1.0

    is_et = meta["leg"] == 0
    assert rho[E.LAYOUT["logk1"]][:, is_et].max() == 0.0
    assert rho[E.LAYOUT["insar_ramp"]][:, meta["leg"] != 2].max() == 0.0


def test_estimator_grid_is_not_the_truth_grid():
    """The inversion must not be able to recover its own discretisation or its own
    conductivity basis."""
    assert C.EST.delr_m > C.TRUTH.delr_m
    assert C.EST.ncol < C.TRUTH.ncol
    k1, _ = __import__("mizan.truth", fromlist=["x"]).logk_fields(C.TRUTH)
    pts = fields.pilot_locations(C.EST, E.NPP)
    assert k1.size > 4 * len(pts)


def test_prior_does_not_use_the_evapotranspiration_data():
    """The abstraction prior is a uniform applied depth over the delineated area. If it
    were derived from the evapotranspiration retrieval, that leg would enter twice."""
    mask_e = fields.upscale_mask(F.pivot_mask(C.TRUTH), 2)
    pr = E.prior(mask_e, C.EST)
    q0 = pr.mean[E.LAYOUT["logq"]].reshape(C.NDIST, C.NYEAR)
    assert np.allclose(q0, q0[:, :1])                    # flat in time
    dmap = C.EST.district_map()
    area = np.array([(mask_e & (dmap == d)).sum() * C.EST.delr_m ** 2 for d in range(C.NDIST)])
    assert np.allclose(10.0 ** q0[:, 0], area * 1.0)     # exactly 1.0 m/yr of depth


# ----------------------------------------------------------------- L2 Kansas guards
def test_unmixing_recovers_a_planted_irrigation_excess():
    """The evapotranspiration leg in Kansas is an unmixing, not a threshold.

    A 1 km pixel over a quarter-section pivot landscape is a mixture, so the estimator
    regresses pixel evapotranspiration on the irrigated fraction of the pixel and reads
    the slope. Plant a known excess and it has to come back. Destroy the pairing between
    the fraction and the pixel and it must not.
    """
    from mizan import ks_data as K

    rng = np.random.default_rng(4)
    nrow, ncol = 20, 30
    region = K.Region(lon0=-102.0, lat0=39.2, nrow=nrow, ncol=ncol, delr_m=2000.0,
                      county=np.zeros((nrow, ncol), dtype=int))
    frac = rng.random((1, nrow, ncol)) ** 2
    dry, excess = 420.0, 300.0
    et = dry + excess * frac + rng.normal(0.0, 12.0, frac.shape)

    vol, se = K.irrigation_et(et, frac, region)
    area = K.irrigated_area(frac, region)
    assert abs(float(vol[0, 0]) / float(area[0, 0]) * 1e3 - excess) < 15.0
    assert float(se[0, 0]) < 0.10 * float(vol[0, 0])

    shuffled = frac.copy()
    rng.shuffle(shuffled[0].reshape(-1))
    bad, _ = K.irrigation_et(et, shuffled[:, ::-1, ::-1], region)
    assert abs(float(bad[0, 0])) < 0.35 * abs(float(vol[0, 0]))


def test_kansas_head_operator_returns_anomalies():
    """Heads are assimilated as anomalies against each well's own record, because the
    casing elevation of a farm well is not a measurement. A level would carry the datum
    error straight into the abstraction estimate."""
    from mizan import ks_run as R

    rng = np.random.default_rng(11)
    nwell, nyear = 7, R.NYEAR
    seen = np.ones((nwell, nyear), dtype=bool)
    seen[3, :4] = False
    heads = rng.normal(900.0, 3.0, (nyear, 5, 5))
    ctx = R.Context(region=None, weight=None, h0=None,
                    well_row=rng.integers(0, 5, nwell),
                    well_col=rng.integers(0, 5, nwell),
                    well_seen=seen, active=None)
    x = np.zeros(R.NPAR)
    out = R.observe(x, heads, ctx)

    flat = np.full((nwell, nyear), np.nan)
    flat[seen] = out["head"]
    per_well = np.nanmean(flat, axis=1)
    assert np.allclose(per_well, 0.0, atol=1e-8)
    assert not np.allclose(heads.mean(), 0.0)


def test_kansas_meters_do_not_reach_the_estimator():
    """The scored quantity is the one thing the run may not look at before scoring."""
    import inspect
    from mizan import ks_run as R

    src = inspect.getsource(R)
    assert "metered_annual" not in src
    assert "wimas_" not in src.replace("ks_fetch", "")

    driver = (Path(__file__).resolve().parents[1] / "scripts" / "11_kansas_run.py"
              ).read_text(encoding="utf-8")
    before, after = driver.split("q_true, meta = K.metered_annual()")
    assert "q_true" not in before
    assert "K.metered_annual" not in after
