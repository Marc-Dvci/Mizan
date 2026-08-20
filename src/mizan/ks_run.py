"""L2 Kansas: the same closure, on real observations, scored against metered pumping.

The L0 experiment predicted that the evapotranspiration and head legs together recover
district-annual abstraction to about five per cent without gravimetry or interferometry.
Northwest Kansas is where that prediction can be tested, because Kansas publishes
per-water-right metered annual pumping and nobody else on Earth does at this density.

What the estimator sees:

* **Actual evapotranspiration** from SSEBop at 1 km, unmixed against the 250 m MIrAD-US
  irrigation map so that the irrigated and dryland components of every pixel are
  separated rather than thresholded, and referenced to the same county's own dryland
  level in the same year, so precipitation and the year's weather cancel;
* **Annual winter water levels** from WIZARD, assimilated as anomalies against each
  well's own record, because the datum of a farm well is not a measurement.

What it never sees: the meters. Those are the scored truth.

Two legs of the four are absent by construction and are declared rather than
approximated. Gravimetry resolves nothing at 16,000 km2, which is smaller than one
mascon. The Ogallala here is unconsolidated and unconfined, so there is no
preconsolidation threshold to cross and no compaction signal to interfere with.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import flopy

from . import inversion as I
from . import ks_data as K

BIN = Path(__file__).resolve().parents[2] / "bin" / "mf6.exe"

NPP = 4                       # pilot points per side
NPILOT = NPP * NPP
NYEAR = K.YEAR1 - K.YEAR0 + 1
NDIST = len(K.COUNTIES)

# Prior applied-irrigation depth over the mapped irrigated area, m/yr. Kansas averages
# close to one acre-foot per acre in the USDA irrigation survey, which is 0.31 m. The
# prior is set at 0.35 with a factor of two either way, and no part of it comes from the
# water-use reports the run is scored against.
PRIOR_DEPTH_M = 0.35
RETURN_FRAC = 0.30            # share of the non-consumed water returning as percolation

HEAD_SIGMA_M = 0.5
ET_REL_SIGMA = 0.15


def _layout() -> dict:
    idx, out = 0, {}
    for name, n in [
        ("logq", NDIST * NYEAR),
        ("eta", NDIST),
        ("logk", NPILOT),
        ("log_sy", 1), ("log_bsat", 1), ("log_rch", 1), ("log_ghb", 1),
    ]:
        out[name] = slice(idx, idx + n)
        idx += n
    out["_n"] = idx
    return out


LAYOUT = _layout()
NPAR = LAYOUT["_n"]


# --------------------------------------------------------------------------- context
@dataclass
class Context:
    """Everything the forward model needs that the inversion never varies."""

    region: K.Region
    weight: np.ndarray            # (NDIST, nrow, ncol) share of county pumping per cell
    h0: np.ndarray                # (nrow, ncol) water table at the start of the record
    well_row: np.ndarray
    well_col: np.ndarray
    well_seen: np.ndarray         # (nwell, NYEAR) bool, where a level exists
    active: np.ndarray            # (nrow, ncol) bool


def interpolate(region: K.Region, lon, lat, val, power: float = 2.0,
                smooth_km: float = 12.0) -> np.ndarray:
    """Smooth inverse-distance surface through scattered point values."""
    LON, LAT = region.centers_lonlat()
    gx, gy = region.to_xy(LON, LAT)
    px, py = region.to_xy(np.asarray(lon), np.asarray(lat))
    d2 = ((gx[..., None] - px) ** 2 + (gy[..., None] - py) ** 2) / 1e6
    w = 1.0 / (d2 + smooth_km ** 2) ** (power / 2.0)
    return (w * np.asarray(val)).sum(axis=-1) / w.sum(axis=-1)


def make_context(region: K.Region, wl: dict, weight: np.ndarray = None) -> Context:
    wells = wl["wells"]
    lon = np.array([w["lon"] for w in wells])
    lat = np.array([w["lat"] for w in wells])
    head = np.array([w["head"] for w in wells])          # (nwell, NYEAR)

    first = np.array([w["head"][np.isfinite(w["head"])][0] for w in wells])
    h0 = interpolate(region, lon, lat, first)
    row = np.array([w["row"] for w in wells])
    col = np.array([w["col"] for w in wells])
    return Context(region=region,
                   weight=K.diversion_weights(region) if weight is None else weight,
                   h0=h0,
                   well_row=row, well_col=col, well_seen=np.isfinite(head),
                   active=region.county >= 0)


# --------------------------------------------------------------------------- prior
@dataclass
class Prior:
    mean: np.ndarray
    sd: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    names: list


def prior(region: K.Region, irr_area: np.ndarray) -> Prior:
    mean = np.zeros(NPAR)
    sd = np.zeros(NPAR)
    lo = np.full(NPAR, -np.inf)
    hi = np.full(NPAR, np.inf)
    names = []

    q0 = np.log10(np.maximum(irr_area.mean(axis=1) * PRIOR_DEPTH_M, 1.0e6))
    mean[LAYOUT["logq"]] = np.repeat(q0, NYEAR)
    sd[LAYOUT["logq"]] = 0.30
    lo[LAYOUT["logq"]] = np.repeat(q0, NYEAR) - 1.2
    hi[LAYOUT["logq"]] = np.repeat(q0, NYEAR) + 1.2
    names += [f"logq_{K.COUNTIES[d]}_{K.YEAR0 + y}" for d in range(NDIST)
              for y in range(NYEAR)]

    # The consumptive fraction as the evapotranspiration product sees it. Bounds are
    # wide enough that a systematic bias in the retrieval lands here rather than in the
    # abstraction, which is what the head leg is there to pin down independently.
    mean[LAYOUT["eta"]] = 0.80
    sd[LAYOUT["eta"]] = 0.12
    lo[LAYOUT["eta"]] = 0.35
    hi[LAYOUT["eta"]] = 1.30
    names += [f"eta_{c}" for c in K.COUNTIES]

    mean[LAYOUT["logk"]] = np.log10(20.0)
    sd[LAYOUT["logk"]] = 0.40
    lo[LAYOUT["logk"]] = np.log10(1.5)
    hi[LAYOUT["logk"]] = np.log10(250.0)
    names += [f"logk_{i}" for i in range(NPILOT)]

    def scalar(key, m, s, a, b, label):
        mean[LAYOUT[key]] = m
        sd[LAYOUT[key]] = s
        lo[LAYOUT[key]] = a
        hi[LAYOUT[key]] = b
        names.append(label)

    scalar("log_sy", np.log10(0.15), 0.16, np.log10(0.04), np.log10(0.32), "log_sy")
    scalar("log_bsat", np.log10(40.0), 0.22, np.log10(12.0), np.log10(140.0), "log_bsat")
    # Recharge under cropland on the Kansas High Plains is reported between about 5 and
    # 35 mm/yr. The bound at 60 keeps the mass balance from being closed by recharge
    # alone, which is the degeneracy this leg has.
    scalar("log_rch", np.log10(20.0), 0.30, np.log10(3.0), np.log10(60.0), "log_rch")
    # Lateral conductance, bounded so the boundary cannot supply the basin either.
    scalar("log_ghb", np.log10(2.0e-2), 0.50, np.log10(1e-4), np.log10(0.2), "log_ghb")
    return Prior(mean, sd, lo, hi, names)


def sample_prior(pr: Prior, ne: int, seed: int = 5) -> np.ndarray:
    """Draw `ne` prior realisations, shape (NPAR, ne).

    County abstraction is drawn with a three-year exponential temporal correlation,
    because an irrigation programme does not resample itself every January.
    """
    rng = np.random.default_rng(seed)
    X = pr.mean[:, None] + pr.sd[:, None] * rng.standard_normal((NPAR, ne))
    yy = np.arange(NYEAR)
    cov = np.exp(-np.abs(yy[:, None] - yy[None, :]) / 3.0)
    L = np.linalg.cholesky(cov + 1e-8 * np.eye(NYEAR))
    sl = LAYOUT["logq"]
    q = pr.mean[sl].reshape(NDIST, NYEAR)[:, :, None] + 0.30 * np.einsum(
        "yz,dzn->dyn", L, rng.standard_normal((NDIST, NYEAR, ne)))
    X[sl] = q.reshape(NDIST * NYEAR, ne)
    return clip(X, pr)


def clip(X: np.ndarray, pr: Prior) -> np.ndarray:
    return np.clip(X, pr.lo[:, None], pr.hi[:, None])


def decode(x: np.ndarray) -> dict:
    return dict(q=10.0 ** x[LAYOUT["logq"]].reshape(NDIST, NYEAR),
                eta=x[LAYOUT["eta"]],
                logk=x[LAYOUT["logk"]],
                sy=10.0 ** float(x[LAYOUT["log_sy"]][0]),
                bsat=10.0 ** float(x[LAYOUT["log_bsat"]][0]),
                rch=10.0 ** float(x[LAYOUT["log_rch"]][0]),
                ghb=10.0 ** float(x[LAYOUT["log_ghb"]][0]))


def pilot_field(logk: np.ndarray, region: K.Region) -> np.ndarray:
    """Bilinear expansion of the pilot points onto the model grid."""
    g = logk.reshape(NPP, NPP)
    r = np.linspace(0, NPP - 1, region.nrow)
    c = np.linspace(0, NPP - 1, region.ncol)
    r0 = np.clip(np.floor(r).astype(int), 0, NPP - 2)
    c0 = np.clip(np.floor(c).astype(int), 0, NPP - 2)
    fr = (r - r0)[:, None]
    fc = (c - c0)[None, :]
    a = g[np.ix_(r0, c0)]
    b = g[np.ix_(r0, c0 + 1)]
    d = g[np.ix_(r0 + 1, c0)]
    e = g[np.ix_(r0 + 1, c0 + 1)]
    return (a * (1 - fr) * (1 - fc) + b * (1 - fr) * fc
            + d * fr * (1 - fc) + e * fr * fc)


# --------------------------------------------------------------------------- model
def build(ws: Path, x: np.ndarray, ctx: Context) -> None:
    p = decode(x)
    reg = ctx.region
    nrow, ncol = reg.nrow, reg.ncol
    botm = ctx.h0 - p["bsat"]
    top = ctx.h0 + 40.0

    sim = flopy.mf6.MFSimulation(sim_name="ks", sim_ws=str(ws), exe_name=str(BIN),
                                 version="mf6", memory_print_option="none")
    perioddata = [(1.0, 1, 1.0)] + [(365.25, 1, 1.0)] * NYEAR
    flopy.mf6.ModflowTdis(sim, nper=NYEAR + 1, time_units="days", perioddata=perioddata)
    flopy.mf6.ModflowIms(sim, complexity="moderate", outer_maximum=80,
                         inner_maximum=250, outer_dvclose=1e-3, inner_dvclose=1e-4,
                         linear_acceleration="bicgstab", print_option="none")
    gwf = flopy.mf6.ModflowGwf(sim, modelname="ks", save_flows=False,
                               newtonoptions="NEWTON UNDER_RELAXATION")

    idomain = ctx.active.astype(int)
    flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=nrow, ncol=ncol,
                            delr=reg.delr_m, delc=reg.delr_m,
                            top=top, botm=botm.reshape(1, nrow, ncol),
                            idomain=idomain.reshape(1, nrow, ncol),
                            length_units="meters")
    flopy.mf6.ModflowGwfic(gwf, strt=ctx.h0.reshape(1, nrow, ncol))
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=(10.0 ** pilot_field(p["logk"], reg)
                                                 ).reshape(1, nrow, ncol))
    flopy.mf6.ModflowGwfsto(gwf, iconvert=1, ss=1e-6, sy=p["sy"],
                            steady_state={0: True},
                            transient={i + 1: True for i in range(NYEAR)})

    flopy.mf6.ModflowGwfrcha(gwf, recharge=p["rch"] / 1000.0 / 365.25)

    ghb = []
    edge = np.zeros((nrow, ncol), dtype=bool)
    edge[0], edge[-1], edge[:, 0], edge[:, -1] = True, True, True, True
    for i, j in zip(*np.nonzero(edge & ctx.active)):
        ghb.append([(0, int(i), int(j)), float(ctx.h0[i, j]), p["ghb"] * reg.delr_m])
    flopy.mf6.ModflowGwfghb(gwf, stress_period_data={0: ghb})

    # Abstraction and return flow, one MODFLOW time series per county so that changing
    # an ensemble member rewrites 6 x 25 numbers instead of a stress-period table.
    cell_w = ctx.weight
    cells = []
    for d in range(NDIST):
        for i, j in zip(*np.nonzero(cell_w[d] > 0)):
            cells.append((d, int(i), int(j), float(cell_w[d][i, j])))
    t0 = np.concatenate([[1.0], 1.0 + np.cumsum(np.full(NYEAR, 365.25))[:-1]])

    def table(vals):
        rows = [tuple([0.0] + [0.0] * NDIST)]
        for k in range(NYEAR):
            rows.append(tuple([float(t0[k])] + [float(v) for v in vals[:, k]]))
        rows.append(tuple([float(t0[-1] + 365.25)] + [0.0] * NDIST))
        return rows

    q_rate = p["q"] / 365.25
    for pname, sign, scale, fn, pre in (("wel_abs", -1.0, np.ones(NDIST), "abs.ts", "q"),
                                        ("wel_ret", +1.0,
                                         RETURN_FRAC * (1.0 - p["eta"]), "ret.ts", "r")):
        rows = [[(0, i, j), f"{pre}{d}", w] for d, i, j, w in cells]
        wel = flopy.mf6.ModflowGwfwel(
            gwf, pname=pname, auxiliary=["wmult"], auxmultname="wmult",
            maxbound=len(rows), stress_period_data={0: rows}, save_flows=False)
        wel.ts.initialize(
            filename=fn, timeseries=table(sign * q_rate * scale[:, None]),
            time_series_namerecord=[f"{pre}{d}" for d in range(NDIST)],
            interpolation_methodrecord=["stepwise"] * NDIST)

    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="ks.hds",
                           saverecord=[("HEAD", "LAST")])
    sim.write_simulation(silent=True)


def run(ws: Path) -> bool:
    import subprocess
    r = subprocess.run([str(BIN)], cwd=str(ws), capture_output=True, text=True)
    return "Normal termination" in r.stdout


def forward(x: np.ndarray, ws: Path, ctx: Context) -> dict:
    """Simulated observation vector for one parameter draw."""
    ws = Path(ws)
    ws.mkdir(parents=True, exist_ok=True)
    build(ws, x, ctx)
    if not run(ws):
        raise RuntimeError("mf6 did not terminate normally")
    hds = flopy.utils.HeadFile(str(ws / "ks.hds"))
    heads = np.array([hds.get_data(kstpkper=k)[0] for k in hds.get_kstpkper()])
    hds.close()
    heads = heads[1:]                                     # drop the steady state
    return observe(x, heads, ctx)


def observe(x: np.ndarray, heads: np.ndarray, ctx: Context) -> dict:
    p = decode(x)
    h = heads[:, ctx.well_row, ctx.well_col].T            # (nwell, NYEAR)
    h = np.where(ctx.well_seen, h, np.nan)
    anom = h - np.nanmean(h, axis=1, keepdims=True)
    return {"et": (p["eta"][:, None] * p["q"]).ravel(),
            "head": anom[ctx.well_seen]}


def head_anomaly(wl: dict, ctx: Context) -> np.ndarray:
    """The measured head anomalies, in the same order the operator returns them."""
    h = np.array([w["head"] for w in wl["wells"]])
    return (h - np.nanmean(h, axis=1, keepdims=True))[ctx.well_seen]


# --------------------------------------------------------------------------- taper
def taper(ctx: Context, radius_km: float = 45.0) -> np.ndarray:
    """Localisation matrix, shape (NPAR, nobs).

    The exact zeros are written directly: an evapotranspiration retrieval over county c
    in year y carries no information about the conductivity field, and no observation
    carries information about pumping that had not happened when it was taken.
    """
    reg = ctx.region
    n_et = NDIST * NYEAR
    wr, wc = ctx.well_row, ctx.well_col
    wy = np.tile(np.arange(NYEAR), (wr.size, 1))[ctx.well_seen]
    wx = (wc[:, None] * np.ones(NYEAR))[ctx.well_seen] * reg.delr_m
    wyy = (wr[:, None] * np.ones(NYEAR))[ctx.well_seen] * reg.delr_m
    nobs = n_et + wy.size

    et_dist = np.repeat(np.arange(NDIST), NYEAR)
    et_year = np.tile(np.arange(NYEAR), NDIST)

    cx = np.zeros(NDIST)
    cy = np.zeros(NDIST)
    for d in range(NDIST):
        rr, cc = np.nonzero(reg.county == d)
        cx[d] = cc.mean() * reg.delr_m
        cy[d] = rr.mean() * reg.delr_m

    rho = np.zeros((NPAR, nobs), dtype=np.float32)
    wgeo = {d: I._gc(np.hypot(wx - cx[d], wyy - cy[d]) / 1000.0, radius_km)
            for d in range(NDIST)}

    sl = LAYOUT["logq"]
    for d in range(NDIST):
        for y in range(NYEAR):
            v = np.zeros(nobs, dtype=np.float32)
            v[:n_et] = ((et_dist == d) & (et_year == y)).astype(np.float32)
            v[n_et:] = wgeo[d] * (wy >= y)
            rho[sl.start + d * NYEAR + y] = v

    sl = LAYOUT["eta"]
    for d in range(NDIST):
        v = np.zeros(nobs, dtype=np.float32)
        v[:n_et] = (et_dist == d).astype(np.float32)
        v[n_et:] = wgeo[d]
        rho[sl.start + d] = v

    px = np.linspace(0, reg.ncol - 1, NPP) * reg.delr_m
    py = np.linspace(0, reg.nrow - 1, NPP) * reg.delr_m
    sl = LAYOUT["logk"]
    k = 0
    for a in py:
        for b in px:
            v = np.zeros(nobs, dtype=np.float32)
            v[n_et:] = I._gc(np.hypot(wx - b, wyy - a) / 1000.0, radius_km + 20.0)
            rho[sl.start + k] = v
            k += 1

    for key in ("log_sy", "log_bsat", "log_rch", "log_ghb"):
        v = np.zeros(nobs, dtype=np.float32)
        v[n_et:] = 1.0
        rho[LAYOUT[key]] = v
    return rho


# --------------------------------------------------------------------------- ensemble
_W: dict = {}


def _init(root, ctx):
    _W["ctx"] = ctx
    _W["ws"] = Path(root) / f"w{os.getpid()}"


def _one(arg):
    i, x = arg
    try:
        return i, forward(x, _W["ws"], _W["ctx"])
    except Exception:
        return i, None


def run_ensemble(X, root, ctx, workers=6):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ne = X.shape[1]
    out = [None] * ne
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(str(root), ctx)) as ex:
        for i, r in ex.map(_one, [(i, X[:, i]) for i in range(ne)], chunksize=1):
            out[i] = r
    ok = np.array([r is not None for r in out])
    if not ok.any():
        raise RuntimeError("every ensemble member failed")
    ref = next(r for r in out if r is not None)
    fill = np.concatenate([ref["et"], ref["head"]])
    D = np.zeros((fill.size, ne))
    for i, r in enumerate(out):
        D[:, i] = np.concatenate([r["et"], r["head"]]) if r is not None else fill
    return D, ok
