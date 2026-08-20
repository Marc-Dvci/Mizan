"""MODFLOW 6 forward model for the Mizan synthetic twin.

One builder serves both the truth and the estimator. Everything the inversion is
allowed to touch arrives through `Params`; everything else is fixed geometry.

Abstraction enters through a WEL package whose rates are supplied by nine MODFLOW
time series, one per district, scaled per cell by an auxiliary multiplier. Changing
the abstraction of a whole ensemble member therefore rewrites 9 x 241 numbers rather
than a stress-period table, which is what keeps an ensemble run cheap.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import flopy

from . import config as C
from . import forcing as F

BIN = Path(__file__).resolve().parents[2] / "bin" / "mf6.exe"

DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=float)


def period_days(nper: int = C.NPER) -> np.ndarray:
    """Length of every transient stress period, days, shape (nper,)."""
    return np.tile(DAYS, int(np.ceil(nper / 12)))[:nper]


def period_start_time(nper: int = C.NPER) -> np.ndarray:
    """Model time at the start of each transient period, days, shape (nper,).

    Time zero is the end of the leading steady-state period.
    """
    d = period_days(nper)
    return np.concatenate([[0.0], np.cumsum(d)[:-1]]) + 1.0


@dataclass
class Params:
    """Everything the estimator may vary."""

    logk1: np.ndarray                 # (nrow, ncol) log10 m/d
    logk2: np.ndarray                 # (nrow, ncol) log10 m/d
    ss_cg: float = C.SS_CG
    sy: float = C.SY
    csub_ssv: float = C.CSUB_SSV
    csub_sse: float = C.CSUB_SSE
    thick_frac: float = C.CSUB_THICK_FRAC
    pcs_offset: float = C.PCS_OFFSET
    recharge_mm_yr: float = C.RECHARGE_MM_YR
    ghb_cond: float = C.GHB_COND
    q_monthly: np.ndarray = field(default=None)   # (NDIST, NPER) m3/d abstraction
    eta: np.ndarray = field(default=None)         # (NDIST,) consumptive fraction


def build(ws: Path, grid: C.Grid, p: Params, mask: np.ndarray,
          insar_cells: np.ndarray | None = None,
          inelastic: bool = False) -> flopy.mf6.MFSimulation:
    """Write a complete MODFLOW 6 simulation for `p` into `ws`.

    With `inelastic`, CSUB also writes the inelastic compaction field, which is the
    storage capacity destroyed permanently. It is read from MODFLOW rather than
    recomputed, so the number reported is the one the solver accounted for.
    """
    ws = Path(ws)
    ws.mkdir(parents=True, exist_ok=True)
    nrow, ncol, delr = grid.nrow, grid.ncol, grid.delr_m
    nper = int(p.q_monthly.shape[1])

    sim = flopy.mf6.MFSimulation(sim_name="mizan", sim_ws=str(ws), exe_name=str(BIN),
                                 version="mf6", memory_print_option="none")
    perioddata = [(1.0, 1, 1.0)] + [(d, 1, 1.0) for d in period_days(nper)]
    flopy.mf6.ModflowTdis(sim, nper=nper + 1, time_units="days", perioddata=perioddata)
    flopy.mf6.ModflowIms(sim, complexity="simple", outer_maximum=60, inner_maximum=200,
                         outer_dvclose=1e-4, inner_dvclose=1e-5, linear_acceleration="bicgstab",
                         relaxation_factor=0.0, print_option="none")
    gwf = flopy.mf6.ModflowGwf(sim, modelname="mizan", save_flows=False,
                               newtonoptions="NEWTON UNDER_RELAXATION")

    flopy.mf6.ModflowGwfdis(gwf, nlay=C.NLAY, nrow=nrow, ncol=ncol, delr=delr, delc=delr,
                            top=C.TOP, botm=C.BOTM, length_units="meters")
    flopy.mf6.ModflowGwfic(gwf, strt=C.H_INIT)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=[1, 0], save_specific_discharge=False,
                            k=[10.0 ** p.logk1, 10.0 ** p.logk2])
    flopy.mf6.ModflowGwfsto(gwf, iconvert=[1, 0], ss=0.0, sy=[p.sy, 0.0],
                            steady_state={0: True},
                            transient={i + 1: True for i in range(nper)})

    # -- compressible interbeds in the upper unit ------------------------------------
    pkgdata = []
    icsub = 0
    for i in range(nrow):
        for j in range(ncol):
            pkgdata.append([icsub, (C.CSUB_LAYER, i, j), "nodelay", C.H_INIT - p.pcs_offset,
                            p.thick_frac, 1.0, p.csub_ssv, p.csub_sse, C.CSUB_THETA,
                            1.0e-6, C.H_INIT])
            icsub += 1
    csub_obs = None
    if insar_cells is not None:
        recs = []
        for n, (i, j) in enumerate(insar_cells):
            recs.append((f"cmp1_{n}", "compaction-cell", (0, int(i), int(j))))
            recs.append((f"cmp2_{n}", "compaction-cell", (1, int(i), int(j))))
        csub_obs = {"csub.obs.csv": recs}
    flopy.mf6.ModflowGwfcsub(
        gwf, head_based=True, initial_preconsolidation_head=True, cell_fraction=True,
        ninterbeds=len(pkgdata), cg_ske_cr=p.ss_cg, cg_theta=C.CSUB_THETA,
        packagedata=pkgdata, save_flows=False, observations=csub_obs,
        compaction_filerecord="mizan.cmp",
        **({"compaction_inelastic_filerecord": "mizan.cmpi"} if inelastic else {}),
    )

    # -- recharge, lateral boundary --------------------------------------------------
    rch = p.recharge_mm_yr / 1000.0 / 365.25
    flopy.mf6.ModflowGwfrcha(gwf, recharge=rch)
    ghb = []
    for i in range(nrow):
        ghb.append([(1, i, 0), C.H_INIT, p.ghb_cond * delr])
        ghb.append([(1, i, ncol - 1), C.H_INIT, p.ghb_cond * delr])
    flopy.mf6.ModflowGwfghb(gwf, stress_period_data={0: ghb})

    # -- abstraction and return flow through per-district time series ----------------
    dmap = grid.district_map()
    weight = np.zeros((nrow, ncol))
    for d in range(C.NDIST):
        sel = mask & (dmap == d)
        if sel.any():
            weight[sel] = 1.0 / sel.sum()

    def ts_table(sign: float, scale: np.ndarray) -> list[tuple]:
        t0 = period_start_time(nper)
        rows = [tuple([0.0] + [0.0] * C.NDIST)]
        vals = sign * p.q_monthly * scale[:, None]
        for k in range(nper):
            rows.append(tuple([float(t0[k])] + [float(v) for v in vals[:, k]]))
        rows.append(tuple([float(t0[-1] + period_days(nper)[-1])] + [0.0] * C.NDIST))
        return rows

    def add_wel(pname: str, sign: float, scale: np.ndarray, tsfile: str, prefix: str):
        spd = []
        for i in range(nrow):
            for j in range(ncol):
                if mask[i, j]:
                    spd.append([(1, i, j), f"{prefix}{dmap[i, j]}", weight[i, j]])
        wel = flopy.mf6.ModflowGwfwel(
            gwf, pname=pname, auxiliary=["wmult"], auxmultname="wmult",
            maxbound=len(spd), stress_period_data={0: spd}, save_flows=False,
        )
        wel.ts.initialize(
            filename=tsfile, timeseries=ts_table(sign, scale),
            time_series_namerecord=[f"{prefix}{d}" for d in range(C.NDIST)],
            interpolation_methodrecord=["stepwise"] * C.NDIST,
        )

    add_wel("wel_abs", -1.0, np.ones(C.NDIST), "abs.ts", "q")
    add_wel("wel_ret", +1.0, C.RETURN_FRAC * (1.0 - p.eta), "ret.ts", "r")

    flopy.mf6.ModflowGwfoc(
        gwf, head_filerecord="mizan.hds", saverecord=[("HEAD", "LAST")],
    )
    sim.write_simulation(silent=True)
    return sim


def run(ws: Path) -> bool:
    """Run MODFLOW 6 in `ws`. Returns True on a normal termination."""
    r = subprocess.run([str(BIN)], cwd=str(ws), capture_output=True, text=True)
    return "Normal termination" in r.stdout
