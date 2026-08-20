# Mizan

**Satellite-constrained abstraction accounting and risk-bounded allocation for unmetered aquifers.**

Four things about a fossil aquifer are visible from orbit: how much water leaves the
surface as crop evapotranspiration, how much mass leaves the basin, how the aquifer
skeleton compacts, and, where wells exist, the head. Each is an estimate on its own.
Together they are an accounting identity, because the aquifer obeys a mass balance and
a stress-strain law, and nothing in current practice forces them to agree.

Mizan makes them agree inside one groundwater-flow and compaction model and solves for
the quantity none of them measures: **the unmetered abstraction field, with calibrated
uncertainty.** It then passes that uncertainty into a chance-constrained allocation
that bounds *permanent* storage loss, and into a data-worth analysis that ranks where
the next meter should go.

---

## What is in this repository

| Layer | Component | Why this one |
|---|---|---|
| Forward physics | **MODFLOW 6** with the **CSUB** package, driven by **flopy** | USGS-authoritative. Elastic and inelastic skeletal storage, preconsolidation head and land subsidence are solved with flow, so irreversibility is a state variable rather than a correlation |
| Inversion | Ensemble smoother with multiple data assimilation, local analysis with R-localisation | Ensemble posterior; parameter, forcing and instrument-nuisance uncertainty propagated together |
| Data worth | Schur complement, cross-checked against **pyEMU** | Standard Bayesian linear data-worth analysis |
| Allocation | **cvxpy**, conditional value at risk in the Rockafellar-Uryasev linear form | Convex, so the tail measure is exact rather than sampled |

Nothing in the physics or the statistics is homemade. The contribution is the coupling:
the closure constraint, the inversion target, and what is done with the posterior.

## Reproducing

```
make setup          # environment and MODFLOW 6 binaries
make all            # truth, ablation grid, allocation, data worth, every figure
make test           # guards, each paired with a corruption that must break it
```

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
src/mizan/allocation.py    response-matrix surrogate and the CVaR allocation
src/mizan/voi.py           Schur-complement data worth
src/mizan/metrics.py       scoring, including interval calibration and resolution
scripts/                   one script per workstream, in order
tests/                     guards, each with its paired corruption
DECISION_LOG.md            what was tried, what failed, and what replaced it
```

## Licence

MIT. Every dependency is permissively licensed or public domain; MODFLOW 6 and PEST
tooling are USGS public domain.
