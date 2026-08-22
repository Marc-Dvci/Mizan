# Method

## 1. The quantity being estimated

A fossil aquifer under irrigation is operated almost blind. Meters cover a fraction of
the wells; where they exist at all they are read annually and often not at all. Every
management instrument in use, a quota, a licence, a well moratorium, is written in
units of abstraction, and abstraction is the one term in the basin's water balance that
nobody measures.

Mizan estimates the **district-annual abstraction field** as a posterior quantity, with
calibrated intervals, from observations that are already in orbit.

## 2. What is observed, and what each observation cannot do alone

| Leg | Instrument | What it constrains | What it cannot resolve |
|---|---|---|---|
| Consumptive use | Satellite actual evapotranspiration over delineated fields | The product of abstraction, the consumptive fraction, and the share applied under an established canopy | Any of the three separately. Water applied before the canopy exists is retrieved as bare-soil evaporation and never enters the account |
| Basin mass | Monthly gravimetry, after removal of soil moisture and canopy storage | Total basin storage change, and therefore the absolute scale of net abstraction | Where inside the basin the water came from. The mascon gain factor and any residual external mass trend enter as nuisances |
| Skeleton response | Interferometric line-of-sight displacement | The response of the aquifer matrix to the change in effective stress, and the split between recoverable and permanent deformation | The absolute datum, which is removed by referencing; and any non-hydraulic deformation, which enters as a nuisance ramp |
| Hydraulic state | Heads at monitoring wells | The local drawdown and its shape in time | The absolute datum at each well, which carries casing and screened-interval error, so heads enter as anomalies about each well's own record |

Each leg is an estimate. None of them is required to agree with the others, and in
current practice none of them does.

## 3. The closure constraint

The aquifer obeys a mass balance and a stress-strain law. Written as a single
statement over one basin and one time step, the four observations are not four
estimates but four projections of one state:

- water leaving the surface as crop evapotranspiration is the consumed part of what was
  applied, and what was applied came from the wells;
- water leaving basin storage is what the wells took, less the part that returns as
  deep percolation, less lateral exchange and recharge;
- the storage that left is the water released from the specific yield of the water
  table and from the compaction of the skeleton;
- that compaction is what the interferometer measures, and its irreversible part is set
  by whether effective stress passed the preconsolidation threshold;
- and the head field is what drives all of it.

Mizan imposes these simultaneously inside **MODFLOW 6 with the CSUB package**, which
solves flow, elastic and inelastic skeletal storage, preconsolidation head and land
subsidence together, and inverts for abstraction as the unknown forcing.

## 4. Forward model

Two layers over a 100 by 100 km basin. An unconfined upper unit carries the water table
through its specific yield; the confined main aquifer beneath it carries the pumping and
the compressible interbeds. Abstraction enters through nine district time series scaled
per cell by an auxiliary multiplier, so changing a whole ensemble member's abstraction
rewrites nine time series rather than a stress-period table.

The compaction formulation is head-based, so the preconsolidation state is expressed as
a **preconsolidation head**, which is directly the threshold the decision layer tracks.

Water released from storage is integrated from the compaction field and the water table
rather than from a cell budget. That identity is checked against MODFLOW's own
volumetric budget in `tests/test_guards.py`, and the check is paired with a corruption
that must make it fail.

## 5. Inverse problem

The unknowns are the district-annual abstraction over twenty years, the consumptive
fraction and pre-canopy share of each district, a pilot-point conductivity field per
layer, the storage and compaction parameters, the preconsolidation offset, recharge and
lateral conductance, and the instrument nuisances.

Inference is by **ensemble smoother with multiple data assimilation**. Three properties
of the implementation matter:

**Localisation is mostly exact.** An evapotranspiration retrieval over a district in a
given year carries no information about the conductivity field, and no observation
carries information about pumping that had not happened when the observation was taken.
Those zeros are written into the taper directly rather than approximated by distance.
Only the head and interferometric blocks use a Gaspari-Cohn distance taper.

**Localisation acts on the error covariance, not on the cross-covariance alone.**
Tapering the parameter-observation cross-covariance while inverting the untapered
innovation covariance leaves the analysis inconsistent, and it diverges when the prior
sits far from the data. Each parameter group is instead updated from the observations
its taper admits, with every admitted observation's error variance divided by its taper
weight.

**The error budget carries model structural error.** The nominal instrument error is
not the error of the inference. A first-stage inversion gives a residual, from which two
quantities are estimated without reference to the truth: the structural component, as
residual variance in excess of nominal, and an independence inflation from the lag-one
autocorrelation of the residual within each site. Treating a persistent structural bias
at a well as two hundred independent monthly measurements is what collapses an ensemble
and produces intervals that do not cover.

## 6. What is reported

- Error of the point estimate against withheld truth, at district-annual resolution.
- Empirical coverage of the 50, 80 and 90 per cent posterior intervals.
- The share of prior variance remaining in every direction of the 180-dimensional
  district-year abstraction vector, so that what the observation set could not resolve
  is published rather than smoothed over.
- The closure residual of every water account. An account that balances exactly is an
  account that has been forced.

## 7. Decision layer

The posterior is not the product. The product is what can be decided with it.

Uniform proportional pumping cuts are evaluated over the twenty-year future horizon for
each selected posterior member. Every reported frontier point is run directly in full
MODFLOW with CSUB and the inelastic switch active. The output is the posterior distribution
of permanent storage-capacity loss at each delivered-water level.

The submitted result is the full-model pumping frontier and its marginal permanent
capacity protected per cubic kilometre not pumped. Response-matrix and CVaR routines are
retained as experimental code, separate from the demonstrated decision product.

## 8. Value of information

Linear data-worth analysis through the Schur complement, cross-checked against pyEMU on
the same inputs. Candidates are a meter on a district, a piezometer at a lattice site,
and a geodetic station at a lattice site. Forward selection produces a ranked siting
plan and a curve of remaining forecast uncertainty against instruments installed.

This is the question a water authority actually faces. Not whether to meter, but which
meter to install first.
