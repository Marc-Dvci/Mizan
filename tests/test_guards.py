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
                    well_seen=seen, active=None, bsat=None,
                    rmul=np.ones((R.NDIST, R.NYEAR)))
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


def test_loco_shrink_never_sees_the_county_it_is_applied_to():
    """The amplitude factor for a county comes from the other counties or it is an oracle.

    The corruption replaces one county's metered anomaly with noise of a different
    amplitude. Its own factor must not move, and at least one other county's must.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "shrink", ROOT / "scripts" / "15_kansas_shrink.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)

    rng = np.random.default_rng(0)
    at = rng.standard_normal((6, 25))
    ah = 2.0 * at + 0.2 * rng.standard_normal((6, 25))
    fac, _ = S.loco(ah, at)
    assert np.all(fac > 0.2) and np.all(fac < 0.9)

    bad = at.copy()
    bad[3] = 40.0 * rng.standard_normal(25)
    fac_bad, _ = S.loco(ah, bad)
    assert fac_bad[3] == fac[3]
    assert np.any(fac_bad[[0, 1, 2, 4, 5]] != fac[[0, 1, 2, 4, 5]])


def test_saturated_thickness_is_the_published_surface_and_not_a_fitted_scalar():
    """The layer base comes from the USGS grid, and the multiplier cannot replace it.

    The corruption flattens the published field to its own mean. The county contrast
    the field carries must vanish, which is the thing the estimated scalar could never
    supply.
    """
    import inspect
    from mizan import ks_data as KD, ks_run as R

    assert "log_bsat" not in inspect.getsource(R)
    assert "saturated_thickness" in inspect.getsource(R)

    src = inspect.getsource(KD.saturated_thickness)
    assert "0.3048" in src, "the published grid is in feet"
    assert "wimas" not in src.lower() and "metered" not in src.lower()

    b = np.array([[10.0, 30.0], [20.0, 40.0]])
    flat = np.full_like(b, b.mean())
    assert b.max() / b.min() > 2.0
    assert flat.max() / flat.min() == 1.0


def test_an_evapotranspiration_account_cannot_exceed_the_reference():
    """A retrieval above reference evapotranspiration is a unit error, not a result.

    The Al Jawf rung compares five published products whose composites carry different
    quantities: a daily rate in one, a period total in another. Reading an eight-day
    total as a rate multiplies the annual account by eight, which is invisible in a
    table of numbers and fatal in a document a jury reads. The ceiling is what settles
    it, so the ceiling is asserted here and asserted to reject the corruption.
    """
    reference = 2025.0
    ceiling = 1.20 * reference

    for account in (622.0, 1436.0, 2110.0):
        assert account <= ceiling

    eight_day_read_as_a_rate = 2110.0 * 8
    assert eight_day_read_as_a_rate > ceiling


def test_the_gravimetric_control_is_reported_as_a_range_not_a_number():
    """Attribution from one control box is a choice, and the choice has to be visible.

    The corruption is a driver that carries a single control: the reported local share
    would then be one number with no spread, and the reader could not see that another
    desert moves it by a factor of ten.
    """
    driver = (Path(__file__).resolve().parents[1] / "scripts" / "20_aljawf.py"
              ).read_text(encoding="utf-8")
    assert "CONTROLS = {" in driver
    assert driver.count("[4") >= 3, "fewer than three control boxes"
    assert "local_share_pct_range" in driver
    assert "CONTROL = [" not in driver


def test_recharge_is_driven_by_observed_precipitation_and_carries_no_water_use():
    """The recharge forcing varies in time, averages one, and never sees a meter.

    Three things can silently break this. The multiplier could stop averaging one, which
    would move the estimated mean recharge off its published prior without saying so. It
    could stop varying, which is the defect it was written to remove. And it could be
    computed from something that carries a water-use term.

    The corruption flattens the precipitation record to its own mean. The multiplier must
    collapse to one everywhere, which is the state the model was in before.
    """
    import inspect
    from mizan import ks_data as KD, ks_run as R

    src = inspect.getsource(KD.recharge_weight) + inspect.getsource(KD.precipitation)
    for banned in ("wimas", "metered", "water right", "pumping"):
        assert banned not in src.lower(), banned

    w = KD.recharge_weight()
    assert w.shape == (len(KD.COUNTIES), KD.YEAR1 - KD.YEAR0 + 1)
    assert np.allclose(w.mean(axis=1), 1.0), "the multiplier must average one per county"
    assert w.min() < 0.75 and w.max() > 1.25, "a forcing this flat is the old defect"

    p = KD.precipitation()
    assert p.max() / p.min() > 1.8, "the precipitation record must carry the range"

    flat = np.full_like(p, p.mean())
    wf = flat / flat.mean(axis=1, keepdims=True)
    assert np.allclose(wf, 1.0), "the corruption must collapse the forcing to one"
    assert not np.allclose(w, 1.0), "the real forcing must not be one"

    # The model has to read it. A per-period recharge dict is the only thing that can
    # carry a year-to-year forcing; the scalar it replaced could not.
    body = inspect.getsource(R.build)
    assert "ctx.rmul" in body and "rspd" in body


def test_the_direction_of_a_declared_change_is_not_reported_as_a_test():
    """Abstraction fell over almost every window pair, so sign is satisfied by a constant.

    The guard is on the driver, not on a number: the verification script must record what
    an estimator that always says "down" would score, so the vacuous statistic cannot be
    quoted on its own. The corruption is a record with a balanced sign, on which the same
    always-down rule scores half and the statistic would have been a test.
    """
    src = (ROOT / "scripts" / "19_verify.py").read_text(encoding="utf-8")
    assert "always_down_scores_on_declared" in src
    assert "declaration_auc_vs_metered_magnitude" in src
    assert "sign_is_not_a_test_on_this_record" in src

    falling = np.array([-12.0, -8.0, -19.0, -3.0, -21.0, 4.0])
    balanced = np.array([-12.0, 8.0, -19.0, 3.0, -21.0, 14.0])
    assert max((falling < 0).mean(), (falling > 0).mean()) > 0.8
    assert max((balanced < 0).mean(), (balanced > 0).mean()) == 0.5


def test_the_mascon_gain_is_only_constrained_by_the_leg_it_multiplies():
    """The gain enters as a prior, and the entry has to be able to say by how much.

    Two things this guards. The gain prior must be overridable from the driver, or the
    sensitivity study cannot be run at all and the assumption stays an assertion. And an
    observing set that carries no gravity leg must return the gain prior untouched, which
    is the statement that the gain is not independently identifiable.

    The corruption is the paired row: a configuration that does carry the gravity leg has
    to move the gain. A test where nothing moves the gain would pass for the wrong reason.
    """
    import numpy as np
    from mizan import estimator as EST

    src = (ROOT / "scripts" / "01_ablation.py").read_text(encoding="utf-8")
    assert "--alpha-sd" in src and "--alpha-mean" in src
    assert "gain_posterior" in src, "the posterior gain has to be recorded per row"

    ia = EST.LAYOUT["grace_alpha"]
    prior_sd = 0.04

    # The twin's gain is offset from the prior on purpose: a sweep run against a truth
    # that agrees with its own prior would measure nothing, because there would be no
    # error for the width of the prior to buy back.
    from mizan import truth as TR
    pr = EST.prior(fields.upscale_mask(F.pivot_mask(C.TRUTH), 2), C.EST)
    mean = float(np.asarray(pr.mean[ia]).ravel()[0])
    assert abs(mean - TR.GRACE_ALPHA) > 1.0 * prior_sd, (
        "the truth gain and the prior mean must differ by more than one prior sd, or "
        "the gain sweep is scoring a prior that was already right")

    def spread(row):
        p = ROOT / "results" / f"posterior_{row}.npz"
        if not p.exists():
            return None
        z = np.load(p)
        ok = z["ok"] if "ok" in z.files else np.ones(z["X"].shape[1], dtype=bool)
        return float(np.asarray(z["X"][ia][:, ok]).ravel().std())

    without = [s for s in (spread(r) for r in ("A", "D", "F")) if s is not None]
    withg = [s for s in (spread(r) for r in ("B", "E", "H")) if s is not None]
    if not without or not withg:
        pytest.skip("ablation posteriors not built")

    # No gravity leg: the posterior is the prior, to within ensemble sampling noise.
    for s in without:
        assert 1.0 - (s / prior_sd) ** 2 < 0.10, (
            "a leg that does not multiply the gain must not constrain it")
    # The corruption: with the gravity leg the gain has to move, or the guard above is
    # passing because nothing in the pipeline touches the gain at all.
    assert max(1.0 - (s / prior_sd) ** 2 for s in withg) > 0.20, (
        "the gravity leg must constrain the gain, or this test proves nothing")


def test_a_degraded_observation_reaches_the_assimilation():
    """An observation the driver alters has to be the observation the row assimilates.

    `run_row` used to rebuild the observation vector from the truth file, so a leg
    degraded on the way in changed the error the estimator was told about and never
    changed the numbers it saw. That is the failure mode where an instrument-noise
    experiment reports an effect it never applied, so the vector is now passed in and
    this guard holds it there.

    The corruption is the arithmetic itself: degrading a leg that already carries its own
    noise means adding the variance difference, not the difference of the standard
    deviations, and a draw checks that the result really lands at the requested error.
    """
    import inspect
    import numpy as np

    sys.path.insert(0, str(ROOT / "scripts"))
    src = (ROOT / "scripts" / "01_ablation.py").read_text(encoding="utf-8")
    body = src[src.index("def run_row("):src.index("def main(")]
    assert "obs_full," in src[src.index("def run_row("):src.index('"""', src.index(
        "def run_row("))], "the observation vector must be an argument of run_row"
    assert 'obs_full = np.concatenate' not in body, (
        "run_row must assimilate the vector it is handed, not rebuild it from the truth")
    assert "--grace-sigma" in src

    # The paired numerical check. A leg generated at s0 and degraded to s carries the
    # variance difference, and the total has to come out at s.
    s0, s = 14.0, 20.4
    rng = np.random.default_rng(0)
    x = s0 * rng.standard_normal(400_000)
    x = x + np.sqrt(s ** 2 - s0 ** 2) * rng.standard_normal(x.shape)
    assert abs(x.std() - s) < 0.1, x.std()
    # And the corruption: adding the difference of the standard deviations does not.
    y = s0 * rng.standard_normal(400_000)
    y = y + (s - s0) * rng.standard_normal(y.shape)
    assert abs(y.std() - s) > 0.5


def test_the_saq_mascon_gain_is_computed_and_not_assumed():
    """The gain the target basin needs is geometry, and geometry can be checked.

    Three limits of the forward model are analytic. A spatially uniform source has to
    return a gain of one over any footprint, because a uniform field survives averaging.
    A footprint made of whole mascons has to return one for any source inside it, which
    is the design rule the entry proposes for the Saq. And the recovered tessellation has
    to look like the published three-degree equal-area design, or the polygons the whole
    calculation rests on are not the product's.

    The corruption is a footprint tighter than one mascon. If that also returned one, the
    forward model would be averaging nothing and every number here would be vacuous.
    """
    import json

    p = ROOT / "results" / "saq_gain.json"
    if not p.exists():
        pytest.skip("saq_gain.json not built")
    d = json.loads(p.read_text(encoding="utf-8"))

    for name, g in d["_checks"]["uniform_source_gain_is_one"].items():
        assert abs(g - 1.0) < 1e-9, (name, g)
    assert abs(d["_checks"]["aligned_footprint_gain_is_one"] - 1.0) < 1e-9
    whole, carrying = d["_checks"]["carrying_mascons_are_whole"]
    assert whole == carrying, "a clipped mascon carries the wrong area and the wrong gain"

    t = d["_tessellation"]
    assert abs(t["median_area_km2"] / t["equal_area_3deg_km2"] - 1.0) < 0.05, t

    tight = d["gain_by_spread"][0]["gain"]["the Al Jawf pivot box"]
    assert tight < 0.5, (
        "a box far smaller than a mascon has to lose most of the signal, or the "
        "averaging in this forward model is not happening at all")


def test_the_external_trend_prior_reaches_the_estimator_and_binds():
    """The other half of the degenerate pair has to be an axis, not a fixed belief.

    Two things this guards. The width of the external mass trend prior must be
    overridable from the driver, or the constraint the entry defends in words cannot be
    scored at all. And a run at a wider prior must actually carry a wider posterior
    trend, which is the check that the override reached the ensemble rather than only
    the metadata.

    The corruption is the paired direction: at the shipped width the posterior trend
    spread has to sit below its own prior. If it did not, the leg would be carrying no
    information about the trend and a widening test would be measuring the prior alone.
    """
    import json

    import numpy as np
    from mizan import estimator as EST

    src = (ROOT / "scripts" / "01_ablation.py").read_text(encoding="utf-8")
    assert "--drift-sd" in src, "the external trend prior has to be overridable"

    def post(tag):
        p = ROOT / "results" / f"posterior_H{tag}.npz"
        if not p.exists():
            return None
        z = np.load(p)
        ok = z["ok"] if "ok" in z.files else np.ones(z["X"].shape[1], dtype=bool)
        return np.asarray(z["X"][EST.LAYOUT["grace_drift"]][:, ok])[0]

    ship, wide = post("_gpub"), post("_dctl")
    if ship is None or wide is None:
        pytest.skip("the drift rows are not built")

    meta = json.loads((ROOT / "results" / "drift_control.json").read_text(
        encoding="utf-8"))["_meta"]
    assert meta["drift_trend_sd"] > 1.0, meta

    # The override reached the ensemble: a wider prior leaves a wider posterior.
    assert wide.std() > 1.5 * ship.std(), (ship.std(), wide.std())
    # The corruption: at the shipped width the data has to bind the trend, or the
    # comparison above is between two priors and says nothing about the estimator.
    assert ship.std() < 1.0, ship.std()


def _lock_pins() -> dict:
    """Every exact pin in the lockfile, keyed by normalised distribution name."""
    pins = {}
    for line in (ROOT / "requirements.lock.txt").read_text(
            encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, version = line.partition("==")
        pins[name.strip().lower().replace("_", "-")] = version.strip()
    return pins


def test_the_environment_the_results_were_produced_in_is_pinned():
    """The documents claim a pinned environment, so there has to be one.

    Version floors in `pyproject.toml` are not a pin: two clones a month apart resolve
    to different builds and the numbers are no longer reproducible from the repository
    alone. `requirements.lock.txt` carries the exact version of every distribution the
    published results were produced with, and this guard holds three properties of it:
    every line is an exact pin, every dependency the project declares is covered, and
    the pinned versions are the ones actually installed here.

    The corruption is a name that is not a dependency. If the coverage check accepted
    it, the check would be passing on a lookup that never fails.
    """
    import re as _re
    from importlib.metadata import PackageNotFoundError, version as installed_version

    lock = ROOT / "requirements.lock.txt"
    assert lock.exists(), "the environment is described as pinned, so the lock is part " \
                          "of the repository"
    pins = _lock_pins()
    assert len(pins) > 20, len(pins)

    for line in lock.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            assert _re.fullmatch(r"[A-Za-z0-9._-]+==[A-Za-z0-9._+!-]+", line), line

    head = lock.read_text(encoding="utf-8")[:600]
    assert "python" in head and "MODFLOW 6" in head, "the interpreter and the solver " \
                                                     "build belong in the header"

    declared = set()
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for m in _re.finditer(r'"([A-Za-z0-9._-]+)\s*(?:[><=!~]=?[^"]*)?"', pyproject):
        token = m.group(1).lower().replace("_", "-")
        if token in ("mizan",) or "." in token and token.count(".") > 1:
            continue
        declared.add(token)
    declared &= {"numpy", "scipy", "pandas", "matplotlib", "flopy", "pyemu", "cvxpy",
                 "tqdm", "pyyaml", "pytest", "earthengine-api"}
    assert len(declared) >= 8, sorted(declared)
    missing = sorted(d for d in declared if d not in pins)
    assert not missing, f"declared but not pinned: {missing}"

    # The corruption: the same lookup has to reject something that is not there.
    assert "mizan-not-a-real-dependency" not in pins

    for name in sorted(declared):
        try:
            here = installed_version(name)
        except PackageNotFoundError:
            continue
        assert here == pins[name], (name, here, pins[name])


def test_the_licence_audit_covers_the_environment_it_claims_to():
    """An audit that lists a subset of the environment is not an audit.

    The table and the lockfile are written by the same run of `make env`, so they have
    to name the same distributions. The corruption is the reverse direction: a name in
    the audit that is not in the lock has to be caught too, so the comparison is not a
    one-sided containment that an empty table would satisfy.
    """
    import re as _re

    audit = (ROOT / "docs" / "LICENCES.md").read_text(encoding="utf-8")
    rows = {m.group(1).strip().lower().replace("_", "-"): m.group(2).strip()
            for m in _re.finditer(r"^\|\s*([A-Za-z0-9._-]+)\s*\|\s*([^|]+)\|",
                                  audit, _re.M)}
    rows.pop("distribution", None)
    rows = {k: v for k, v in rows.items() if set(k) != {"-"}}
    pins = _lock_pins()
    assert len(rows) > 20, len(rows)
    assert not sorted(set(pins) - set(rows)), sorted(set(pins) - set(rows))[:5]
    assert not sorted(set(rows) - set(pins)), sorted(set(rows) - set(pins))[:5]

    # Nothing may be left without a licence, and no strong copyleft may appear, because
    # the repository ships under MIT and the proposal says so.
    # The licence column itself, rather than the prose around it: the verdict
    # sentence names the licences it rules out, so scanning the whole file would
    # match its own conclusion.
    strong = _re.compile(r"\b(GPL-[23]|AGPL|LGPL|GNU General Public)", _re.I)
    for name, cell in rows.items():
        assert cell and "not declared" not in cell, (name, cell)
        assert not strong.search(cell), (name, cell)
    # The corruption: the same pattern has to fire on a licence that is copyleft.
    assert strong.search("GPL-3.0-or-later")


def test_the_results_table_is_written_inside_the_repository():
    """The documents send a reader to `RESULTS.md`, so `make report` has to write it.

    The target used to redirect above the repository root, which put the file the
    submission cites outside the clone a reader gets. The corruption is the redirect
    itself: a target that writes anywhere above the root fails this.
    """
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    body = makefile[makefile.index("\nreport:"):]
    body = body[:body.index("\n\n")]
    assert "07_report.py > RESULTS.md" in body, body
    assert "../" not in body, body

    out = ROOT / "RESULTS.md"
    assert out.exists(), "make report"
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Results")
    assert "make all" in text and "Mm" in text
