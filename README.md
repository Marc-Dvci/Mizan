# Mizan

**Satellite-constrained abstraction accounting and permanent-storage decision support for unmetered aquifers.**

Four things about a fossil aquifer are visible from orbit: how much water leaves the
surface as crop evapotranspiration, how much mass leaves the basin, how the aquifer
skeleton compacts, and, where wells exist, the head. Each is an estimate on its own.
Together they are an accounting identity, because the aquifer obeys a mass balance and
a stress-strain law, and nothing in current practice forces them to agree.

Mizan makes them agree inside one groundwater-flow and compaction model and solves for
the quantity none of them measures: **the unmetered abstraction field, with calibrated
uncertainty.** It then runs a full-MODFLOW frontier from delivered water to *permanent*
storage-capacity loss, and a data-worth analysis that ranks where the next meter should
go.

---

## What is in this repository

| Layer | Component | Why this one |
|---|---|---|
| Forward physics | **MODFLOW 6** with the **CSUB** package, driven by **flopy** | USGS-authoritative. Elastic and inelastic skeletal storage, preconsolidation head and land subsidence are solved with flow, so irreversibility is a state variable rather than a correlation |
| Inversion | Ensemble smoother with multiple data assimilation, local analysis with R-localisation | Ensemble posterior; parameter, forcing and instrument-nuisance uncertainty propagated together |
| Data worth | Schur complement, cross-checked against **pyEMU** | Standard Bayesian linear data-worth analysis |
| Decision support | **MODFLOW 6 + CSUB** over the posterior pumping frontier | Every submitted frontier point is evaluated in the nonlinear simulator; experimental CVaR allocation code is retained separately |

Nothing in the physics or the statistics is homemade. The contribution is the coupling:
the closure constraint, the inversion target, and what is done with the posterior.

## Reproducing

```
make setup          # the locked environment and the MODFLOW 6 binaries
make all            # truth, ablation grid, allocation, data worth, every figure
make test           # guards, each paired with a corruption that must break it
make robustness     # repeat the headline rows across prior ensembles and truths
make kansas-data    # retrieve the public Kansas records, no account needed
make kansas         # the L2 rung, scored against metered pumping
make kansas-score   # the anomaly, resolution and amplitude scoring on top of it
make aljawf         # the L3 rung, read live from Earth Engine
make env            # rewrite the lock and the licence audit from what is installed
make gain           # what the mascon gain assumption costs, on an axis
make saq-gain       # what the mascon gain is on the target basin, from its geometry
make drift          # what the external mass trend prior costs, against the L3 controls
```

`make aljawf` and `make saq-gain` are the only targets that need an account. Run `earthengine authenticate`
once and set `EARTHENGINE_PROJECT` to your own cloud project; registration is free and
every asset the rung reads is public.

`make all` writes every number and every figure that appears in the submission into
`results/` and `figures/`. Nothing in the submission comes from anywhere else.

## The experiment

`L0` is a blind synthetic experiment. A 100 by 100 km fossil-aquifer irrigation basin
is simulated on a 1 km grid over twenty years, with nine management districts, 1,943
km2 of centre pivots, 2.8 m/yr of peak head decline and 4.2 cm/yr of peak subsidence.
Everything about it is then hidden from the estimator: the conductivity field, every
storage parameter, the preconsolidation head, the recharge, the consumptive fraction of
each district, the share of irrigation applied before the canopy exists, and the
district pumping itself.

The estimator works on a **2 km grid with a 5 by 5 pilot-point basis per layer**, so it
can recover neither its own discretisation nor its own conductivity basis. Synthetic
observations carry the error models the real instruments have, including the ones that
are themselves unknown: gravity leakage and an external mass signal, a non-hydraulic
interferometric ramp, and an evapotranspiration retrieval that is blind to pre-canopy
irrigation.

The scored quantity is **district-annual abstraction**, which is what a regulator can
act on. The baseline is the published open-loop account, consumptive use divided by an
assumed efficiency, scored both at its published constant and at the single global
efficiency fitted against the answer, which no practitioner would have.

## Layout

```
src/mizan/config.py        domain, time, physical constants, the two grids
src/mizan/forcing.py       centre-pivot geometry and the abstraction forcing
src/mizan/model.py         the MODFLOW 6 + CSUB forward model
src/mizan/observations.py  the four observation operators and their error models
src/mizan/truth.py         the hidden reality of the L0 experiment
src/mizan/estimator.py     what the estimator may vary, and what it believes a priori
src/mizan/inversion.py     ES-MDA, localisation, and the total error budget
src/mizan/allocation.py    pumping-frontier helpers and experimental CVaR allocation
src/mizan/voi.py           Schur-complement data worth
src/mizan/metrics.py       scoring, including interval calibration and resolution
src/mizan/ks_fetch.py      retrieval of the public Kansas records
src/mizan/ks_data.py       the Kansas region, its observations and its withheld truth
src/mizan/ks_run.py        the Kansas forward model, prior, operators and localisation
scripts/                   one script per workstream, in order
tests/                     guards, each with its paired corruption
RESULTS.md                 every generated number, written by `make report`
requirements.lock.txt      the environment the published results came from
docs/LICENCES.md           the licence of every distribution in that environment
DECISION_LOG.md            what was tried, what failed, and what replaced it
```

## The rungs

| Rung | What it tests | Where |
|---|---|---|
| **L0** | Whether the inverse problem is identifiable at all, against a truth the estimator never sees | `scripts/00_truth.py` to `scripts/07_report.py` |
| **L0 robustness** | Whether the result survives three independent prior ensembles, and whether it survives removing the planted spread in the consumptive fraction | `make robustness` |
| **L2 Kansas** | Whether it recovers **real metered abstraction**, in six counties of the Northwest Kansas groundwater management district over the Ogallala, 2000 to 2024 | `make kansas-data && make kansas && make kansas-score` |
| **L3 Al Jawf** | How far apart the published instruments are over the target basin, from public data. Nothing here is fitted and nothing is scored | `make aljawf` |
| **The mascon gain** | Whether the gravity leg's gain is identifiable at all, and what its prior costs the absolute scale of the account | `make gain` |
| **The gain on the Saq** | What that gain is on the target basin, computed from the mascon polygons recovered from the published product rather than assumed | `make saq-gain` |
| **The external mass trend** | What the other nuisance in the gravity operator costs when its prior is widened to the scale the L3 control boxes measure | `make drift` |

L2 is scored against per-water-right metered annual pumping published by the Kansas
Department of Agriculture through WIMAS. The meters are read once, at the end, to score.
Nothing upstream of the scoring touches them. The observations the estimator is allowed
to see are SSEBop actual evapotranspiration unmixed against the MIrAD-US irrigation map,
and WIZARD annual water levels. Two of the four legs are absent there and are declared
rather than approximated: the block is smaller than one gravimetric mascon, and the
Ogallala is unconsolidated and unconfined, so there is no preconsolidation threshold to
cross.

The published configuration is `_v3`: a per-site error budget, and a layer base taken
from the USGS saturated-thickness grid rather than estimated. The first configuration
ran on an estimated thickness of 79 m where that published surface gives 20.4 m over the
same counties, which is a falsification against an independent observation of geometry
rather than against a score. `_v3p` is the pooled-budget sensitivity and both stay
runnable. `DECISION_LOG.md` records that the change was made after the first score was
known.

## Data

Every source is a public record. Only the Earth Engine rung needs an account, and it is
free.

| Source | What it provides | Held by |
|---|---|---|
| **WIMAS** | Per-water-right metered annual pumping, the withheld truth for L2 | Kansas Department of Agriculture, Division of Water Resources, served by the Kansas Geological Survey |
| **WIZARD** | Annual winter water levels at 449 wells | Kansas Geological Survey |
| **SSEBop** | Annual actual evapotranspiration, CONUS, 1 km | USGS EROS |
| **MIrAD-US** | Irrigated agriculture, CONUS, 250 m, 2002 to 2017 | USGS EROS |
| **hp_satthk09** | Saturated thickness of the High Plains aquifer, 2009, 500 m. The base of the L2 layer is taken from it rather than estimated | USGS, published with SIR 2012-5177 |
| **MODIS, PML V2, WaPOR v2 and v3, TerraClimate, GRACE and GRACE-FO** | The L3 rung, read live from Earth Engine over Al Jawf and four control deserts | NASA, FAO, CSIRO, University of Idaho, JPL and GFZ |

The retrieved files land in `data/kansas/` and are excluded from version control, so a
clone reproduces them rather than carrying them.

## Licence

MIT. Every dependency is permissively licensed or public domain; MODFLOW 6 and PEST
tooling are USGS public domain.
