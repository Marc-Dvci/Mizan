# Decision log

Append-only. Every entry records something that was tried, what it produced, and what
was done about it. Entries that record a failure stay in.

---

### Aquifer configuration, first attempt: fully confined

Built the L0 twin as two confined layers with a coarse-grained specific storage of
1e-6 /m and compressible interbeds in the upper unit.

**Result: rejected.** Peak head decline reached 2,427 m against an aquifer bottom at
-400 m, and 21 of the 31 km3 abstracted arrived as lateral inflow through the boundary.
A confined storage coefficient of order 1e-3 cannot supply 0.16 m/yr of equivalent
water depth for twenty years; a real basin supplies it from an unconfined or
semi-confined water table over a much larger contributing area.

**Replaced by:** an unconfined upper unit with a specific yield of 0.08, a confined
main aquifer beneath it, and the compressible interbeds moved inside the pumped
confined unit, which is where the Central Valley and Saq analogues put them. Head
decline is now 2.8 m/yr peak and subsidence 4.2 cm/yr peak, both inside the range
published for Saudi agricultural areas over the fossil aquifer.

**Cost accepted:** the flow problem is no longer exactly linear, so superposition in
the allocation layer is an approximation rather than an identity. That is why every
candidate policy is re-run in full MODFLOW and the surrogate-to-simulator discrepancy
is reported rather than assumed away.

---

### Storage identity, verified rather than assumed

The gravity operator needs total basin storage change. Rather than write the cell
budget for every stress period, which produced a 121 MB file per run and dominated
the runtime, storage change is integrated from the compaction array and the water
table.

**Checked against MODFLOW's own volumetric budget on the truth run:** specific-yield
term 26.1091 km3 against 26.1090 km3 listed, skeletal and coarse-grained term 2.95058
km3 against 2.9506 km3 listed. The water-compressibility term, 0.221 km3 or 0.76 per
cent of the total, is not represented in the operator; it is twenty times below the
gravity noise floor. The guard is in `tests/test_guards.py` and is paired with a
corruption that must make it fail.

---

### Ensemble smoother localisation, first attempt: taper the cross-covariance

Applied a physically motivated taper to the parameter-observation cross-covariance
only: exact zeros where an evapotranspiration retrieval cannot inform a conductivity
field, exact zeros where an observation predates the pumping it would otherwise
inform, and a Gaspari-Cohn distance taper elsewhere.

**Result: rejected. It diverged.** On the evapotranspiration leg alone, the objective
function rose from 1,414 to 23,936 across four assimilations and the abstraction
estimate left the physical box. Tapering the cross-covariance leaves the analysis
inconsistent with the innovation covariance it was solved against, and the retained
components over-correct to compensate for the ones that were zeroed.

**Replaced by:** local analysis with R-localisation. Each parameter group is updated
from the observations its taper admits, with every admitted observation's error
variance divided by its taper weight. On the same test the objective fell to 0.93 and
the error against withheld truth fell from 32.9 Mm3/yr with no localisation to 11.1
Mm3/yr with it.

---

### Ensemble smoother stability

The first working version still left the parameter box on the first assimilation,
because a prior drawn without reference to the data sits tens of standard deviations
from it.

**Fixed by three standard measures, all kept:** admissible bounds on every parameter
that are physical statements rather than numerical guards, a cap of two prior standard
deviations on any single parameter's step, and a decreasing inflation schedule whose
reciprocals sum to one, so the first assimilation is damped by a factor of fifteen.

---

### Error budget, first attempt: nominal instrument error only

Ran the full four-leg inversion weighting every observation at its instrument error:
6 mm for interferometric displacement, 0.5 m for heads, 14 mm equivalent water height
for gravity, 12 per cent for evapotranspiration.

**Result: rejected.** District-annual abstraction error came out at 23.9 Mm3/yr,
worse than the 16.7 Mm3/yr of the published open-loop account, and the ninety per cent
posterior intervals covered 15 per cent of the truth. The aquifer parameters were
nonetheless recovered well: preconsolidation offset 11.93 m against 12.0 m true,
inelastic skeletal storage 3.76e-4 against 3.5e-4 true. The point estimate and the
intervals were both wrong for the same reason, which is that the likelihood was far too
sharp.

**Diagnosed:** running the coarse estimator model with the true parameters upscaled
onto its own grid leaves a residual of 10.6 mm against a 6 mm nominal interferometric
error and 0.97 m against a 0.5 m nominal head error. The gravity leg has essentially no
structural error. Treating a persistent structural bias at a well as two hundred
independent monthly measurements is what collapsed the ensemble.

**Replaced by:** a two-stage total error. A short first-stage inversion on the full
observation set gives the residual, from which two quantities are estimated without
reference to the truth: the structural component, as the residual variance in excess of
the nominal, and an independence inflation from the lag-one autocorrelation of the
residual within each site. The resulting budget is a property of the forward model, so
the same budget is applied to every row of the ablation grid.

---

### Guard against the inverse crime

The truth runs on a 1 km grid with a dense exponential-covariance conductivity field.
The estimator runs on a 2 km grid and may only move a 5 by 5 array of pilot points per
layer. Neither the discretisation nor the conductivity basis is recoverable by the
inversion. `tests/test_guards.py` asserts both.

---

### The abstraction prior does not use the evapotranspiration data

The prior on district-annual abstraction is a uniform 1.0 m/yr applied depth over the
delineated irrigated area, with a factor-of-two spread. Seeding it from the
evapotranspiration retrieval would have been the natural choice and would have made
that leg enter the inference twice. Asserted in `tests/test_guards.py`.

---

### Assimilating heads and interferometry as levels

Both legs were first assimilated as absolute values: head at a well, line-of-sight
displacement at a pixel referenced to a stable pixel.

**Result: rejected.** Structural error on the coarse estimator grid was 0.97 m against a
0.5 m instrument error on heads, and 10.6 mm against 6 mm on displacement, so on both
legs the model could not reach the noise floor.

**Replaced by:** each well and each pixel referred to the mean of its own record.
Structural error fell to 0.44 m and 5.2 mm, both below the instrument noise. It is also
the physically correct operator. A regional model cannot be held to the absolute head at
a well whose casing elevation and screened interval carry their own error, and an
interferogram has no absolute datum at all.

---

### Gravity nuisance parameters, first attempt: a free gain and a free linear drift

The gravity operator carried a mascon gain factor with a prior of 0.88 plus or minus
0.08, and a three-term external mass signal: linear trend, annual sine and annual
cosine, all freely estimated.

**Result: rejected.** The inversion fitted every leg to its noise floor and still
overestimated basin abstraction by 23 per cent, with the gain riding its lower bound in
46 per cent of ensemble members and the fitted trend coming out at the wrong sign.
Basin storage falls close to linearly over the record, so a multiplicative gain and an
additive linear trend multiply nearly the same function of time: between them they
absorb the absolute scale of the answer while leaving the fit untouched.

**Replaced by:** the gain treated as computed rather than fitted, 0.85 plus or minus
0.04 inside a 0.75 to 0.97 box, and the external trend constrained to plus or minus 1.0
mm/yr. The justification is physical: in a hyper-arid basin with no surface water and no
snow, the trend in total water storage is the trend in groundwater.

**Addendum, 21 August 2026: the 23 per cent is history and is not reproducible.** It was
measured on the first version of the estimator, before the R-localised analysis, the
step cap and the relaxation to prior spread, and no results file was kept. `make gain`
now runs the release deliberately, at a wider prior than this entry ever shipped (gain sd
0.5 across the box, external trend sd 3.0), and it does not reproduce the figure: the
scale moves 3.8 points rather than 23, mean absolute error rises 47 per cent, and the
gain itself comes back within 0.005 of the twin's true 0.78 while the external trend
takes up the error instead. The mechanism the entry describes is confirmed and the
magnitude is not, so 23 per cent is retired from the judge-facing documents and replaced
by the measured numbers in section 5.5 of the technical note.

**What this established, and it is the finding the entry is built on:** the satellites
determine the *pattern* of abstraction in space and time; the *absolute scale* rests on
the gravity leg alone. That is why the value-of-information layer exists and why the
pilot path treats existing well meters as the scale calibration rather than as a
nice-to-have.

---

### Ensemble collapse

Even with the scale degeneracy removed, the localised smoother under-dispersed: 90 per
cent intervals covered a small fraction of the truth after four assimilations, and the
point estimate drifted away from the truth while the fit continued to improve.

**Fixed by relaxation to prior spread** at 0.7, the standard treatment. Coverage of the
90 per cent interval moved to 0.89 against a nominal 0.90, and the point error fell
monotonically instead of turning around. `tests/test_guards.py` asserts that the
relaxation widens a collapsed ensemble and leaves its mean where the analysis put it.

---

### Response-matrix surrogate, first attempt: district resolution

The allocation surrogate averaged head over each whole district, then applied the
positive part of the preconsolidation exceedance to that average.

**Result: rejected, and the verification step is what caught it.** The surrogate
predicted 723 Mm3 of permanent storage loss at the 90 per cent tail where full MODFLOW,
run on the same policy with the inelastic switch active, produced 3,441 Mm3. The policy
the surrogate selected came out **1.8 per cent worse than the uniform-quota baseline**
when actually simulated.

The cause is Jensen's inequality. Inelastic compaction is a positive part, so averaging
head over an area before applying it understates the loss, and drawdown inside one
district ranges from nothing to tens of metres, so the average hides every cell that is
already deep in the inelastic regime.

**Replaced by:** the same superposition surrogate on drawdown-stratified zones, five per
district, built from the same nine pulse runs at no extra cost. The convexity error
inside a stratum is small because head inside a stratum is close to uniform.

**Kept regardless:** every candidate policy is re-run in full MODFLOW and the
surrogate-to-simulator discrepancy is reported. Optimising on a linearisation and
verifying on the simulator is the defensible order, and here the verification was the
only thing standing between a wrong surrogate and a claimed result.

---

### Time invariance of the response matrix

The matrix is built from a pulse in the first horizon year and reused at every later
lag. Measured rather than assumed, by pulsing each district in year ten and comparing
against what superposition predicts: **mean worst-case error 0.4 per cent of the pulse
response** across three posterior members. The aquifer is not exactly linear, since the
upper unit is unconfined and inelastic compaction switches on at a threshold, but over
this range the superposition assumption holds.

---

### The ablation grid, and what it settled

Fourteen inversions of the same basin, differing only in which observations the
estimator was allowed to see, all sharing one prior ensemble of 250 members and eight
assimilations. Three results were not anticipated when the grid was designed.

**Every leg alone fails.** Heads alone 90.99 Mm3/yr, gravity alone 93.48, deformation
alone 146.35, against 16.66 for the published open-loop account. No sensor in this
system is the invention.

**The evapotranspiration leg is removable.** Row G, heads with gravity and deformation
and no consumptive-use retrieval at all, reaches 8.46 Mm3/yr and 6.7 per cent. That
matters more than the headline number, because the reasonable objection to this work is
that it is López Valencia et al. with extra sensors bolted on. It is not: the closure
stands without their leg, and beats their published method without it.

**Meters did not help.** A complete metering network on every district, read to 8 per
cent, gives 12.26 Mm3/yr on its own. Added to the four satellite legs it gives 6.96 with
one district metered and 7.81 with three, against 6.72 with none. At this level of
metering accuracy an annual meter read carries less information about a district-year
than the closure already has, and adding it perturbs the fit rather than sharpening it.

That result is entirely conditional on the assumed 8 per cent read accuracy and is
reported as such. It is also the counterweight to the value-of-information
narrative, and it is kept in rather than dropped.

**Differences below about 1 Mm3/yr are not resolved by a single ensemble.** The same
configuration run with a nine-step schedule gave 6.61 against 6.72 for the eight-step
grid row. Nothing in the ranking of rows separated by less than that is claimed.

---

### Testing the enforcement claim instead of asserting it

The claim that "a district whose books do not balance is a district to inspect" was in
the plan from the start, as an assertion. It was tested by planting one.

A withdrawal of 40 Mm3/yr was added to district 3 in the truth, with no crop and
therefore no canopy, so it produces no signature in any evapotranspiration product ever
made. It is 39 per cent of that district's abstraction. The evapotranspiration
observations handed to the estimator were left unchanged.

**Result.** The open-loop account attributes 44 per cent of that district's true
abstraction, which is what being blind to the withdrawal looks like. The closure
attributes 84 per cent. On the ratio of the closure estimate to what consumptive use can
explain, the district sits at 1.90 against 0.97 to 1.17 for the other eight, which is
11.4 standard deviations clear.

**What is not claimed.** The closure recovers 84 per cent of that district's abstraction,
not all of it. The estimator's parameterisation ties abstraction to consumptive use
through a bounded efficiency and pre-canopy share, so a withdrawal this large cannot be
fully absorbed by those parameters. Detection is unambiguous; full recovery of the hidden
volume is not, and would need a per-district non-agricultural abstraction term that this
version does not carry.

### Resolution metric, first version: a signed sum over directions

The share of prior variance surviving in each direction of the 180-dimensional
district-year abstraction vector is the right diagnostic, and it is reported. The
summary built on top of it was not. Summing `1 - ratio` over all directions gave
`gravity only` an "effective dimension" of **-64.7** and `deformation only` **-256.5**,
which is not a dimension.

The cause is real and worth reporting rather than hiding. Relaxation to prior spread,
and the localised analysis, leave some directions with more posterior variance than
prior variance. A direction that widened has learned nothing from the data; it has not
learned a negative amount. The summary is now `sum of max(0, 1 - ratio)`, the number of
directions the data actually constrained, and the count of widened directions is
published alongside it.

The reading changes with it. The coupled closure constrains **135.4** of 180 directions
and widens 20. Gravity alone constrains **37.1** and widens 101, which is the same
statement the ablation makes in Mm³/yr, arriving from the covariance instead.

Reported by `scripts/07_report.py` from `results/posterior_*.npz`, so the number comes
from the saved posterior rather than from a field written at run time.

### L2 Kansas saturated thickness, first version: a global scalar, estimated

The first Kansas configuration estimated one saturated thickness for the whole block,
with a log-uniform prior from 20 m to 140 m centred on 45 m, and settled at **78.7 m**.

**Rejected against a published surface, not against a score.** The USGS High Plains
saturated-thickness grid `hp_satthk09`, 500 m, published in feet with SIR 2012-5177,
sampled onto the model grid, gives a block mean of **20.4 m** and county means of 19.7,
16.6, 13.2, 29.4, 23.6 and 19.1 m. Five of the six counties sit at or below the prior's
lower bound, so the prior did not merely miss the published value, it excluded it. That
grid is an observation of the aquifer's geometry into which no water-use report enters,
so the model was falsifiable against it before any meter was opened.

The mechanism is visible in the scores. Too much transmissivity spreads drawdown too far
and too thin, which is the condition under which a head record fits well and constrains
nothing. Sheridan county carried the entire level gap at 52.4 Mm³/yr against a metered
91.9, with 90 per cent coverage of 0.40 where every other county was above 1.2, and a
forward run at the posterior mean fitted Sheridan's heads to 0.83 m rms with no residual
trend: the head data there never constrained the abstraction, so the estimate was free to
move.

**Replaced by:** the layer base is the published field, sampled and converted, times one
estimated multiplier with a prior centred on 1.0 and bounds 0.6 and 2.0, floored at 8 m
per cell so the Newton solve survives the margin of the mapped aquifer. The parameter
count is unchanged, so the comparison against the previous run is clean. The inversion
chose **0.90**, so the head record is consistent with the published surface rather than
pulling away from it. Sheridan returns 109.4 at coverage 0.96.

**What it did not buy, recorded for the same reason.** Cheyenne moved the other
way, 53.8 to 36.6 against a metered 59.9, and the heads-only row puts it at 33.4, so the
head leg now drags Cheyenne as it used to drag Sheridan. Mean absolute error improves
17.65 to 16.57 and relative error worsens 23.3 to 25.4 per cent. No closure configuration
beats mapped irrigated area times one acre-foot per acre at 14.78 Mm³/yr.

**This change was made after the first Kansas score was known.** It is recorded here for
that reason. The decision was taken against the published surface rather than against the
score, both configurations remain in the repository and both remain runnable, and with
six counties there is no untouched data left to size a further effect on. Nothing after
this was tuned.

### Kansas error budget: per site rather than pooled

A structural error term read from the converged residual of a first, instrument-weighted
assimilation, and applied per observation site rather than pooled across a leg.

**Result: kept, and the pooled variant kept beside it.** Under the published thickness
the per-site budget scores 16.57 Mm³/yr at 25.4 per cent with 95 per cent coverage; the
pooled budget scores 16.92 at 26.5 per cent with 97 per cent. Both are reported. Neither
was selected against the meters, and the choice between them changes no claim in the
submission.

An earlier version of the same idea, run on the *estimated* thickness, made things worse:
mean absolute error 17.65 to 18.82 and Sheridan's coverage 0.40 to 0.32, because inflating
the evapotranspiration leg by 22 per cent against the head leg's 13 raised the head leg's
weight and the head leg was what pulled Sheridan down. The mechanism was right and the
model underneath it was wrong. That is recorded because it is the reason the thickness was
checked at all.

### Kansas interannual amplitude: an oracle shrink, replaced by leave-one-county-out

The closure's county-year anomaly correlates with the metered anomaly at 0.64. Under a
mean-absolute-error metric a correlation below one puts the amplitude that minimises the
error below the estimate's own, so a single scalar improves the anomaly score.

**Rejected as first written.** The scalar was fitted against all six counties at once,
which is an oracle: it is fitted on the same meters it is then scored against, and it
cannot be reported as a result.

**Replaced by:** the same scalar fitted leave-one-county-out, so every county's factor
comes from the other five and no county enters its own fit. The factor runs 0.68 to 0.75
across the six folds, the held-out anomaly skill is +0.18 against an oracle +0.18, and
both columns are published side by side so the distance between them is visible rather
than asserted.

### Kansas ensemble convergence under the published thickness

Putting the true thickness in costs part of the prior ensemble: thin low-storage members
dewater and the Newton solve fails on them. Measured on 80 members by
`scripts/16_kansas_convergence.py`, 27 failed.

**Result: reported rather than absorbed.** The failed members sit 0.07 standard
deviations from the converged ones on block abstraction, so the scored quantity is
unaffected. They sit 0.77 away on specific yield and 0.46 on the thickness multiplier.
The truncation is selective in exactly those two directions, so the posterior specific
yield of 0.249, which sits above the range published for the Kansas High Plains, is
reported as an **upper bound rather than as a retrieval**, in the technical note and in
the generated results.

### L3 Al Jawf, first version: prose with no script behind it

An earlier assessment reported two Al Jawf probes as numbers in a document with no
script, no results file and no way to re-run them.

**Rejected, and it did not reproduce.** It reported PML V2 at 162 mm/yr for 2021 from a
collection that ends in 2020; a crop-coefficient account at 958 mm/yr against 1,436 here;
and a Rub' al Khali control at -8.06 cm/decade against -3.32 here, which inverts the
conclusion from "at most a quarter of the trend is local" to "between 7 and 71 per cent,
depending on the control".

**Replaced by:** `scripts/20_aljawf.py` and `scripts/21_aljawf_figure.py`, everything read
live from Earth Engine, written to `results/aljawf.json`, with the delineation validated
against the published one (2,541 km² against 2,494 km² at 30 m) and its threshold
sensitivity published alongside. Two guards are paired with it: an evapotranspiration
account above 1.2 times reference evapotranspiration fails a conservative plausibility
screen and is rejected, and the driver is asserted to carry more than one control box.

### L2 resolution metric read against zero, replaced by a measured null

The number of directions the data constrained was being read against zero, on the
assumption that a prior ensemble with no data assimilated constrains none.

**Rejected.** `scripts/13_null.py` assimilates nothing and measures what two independent
prior ensembles constrain by themselves: **58.3** of 180 directions at L0 and **44.8** of
150 at L2, from relaxation and localisation alone. Every row was being credited with that
much for free.

**Replaced by:** the null row is published in both tables and every reading is stated
above it. Null-corrected, the L0 closure constrains +77.1 and gravity alone comes out at
**-21.2**, below what nothing does. At L2 the head leg carries the block at +31.6, the
closure sits at +28.8, and the evapotranspiration leg alone at -3.9 is also below the
null: a 1 km retrieval over a block that is 14 per cent irrigated carries a structural
error large enough to cancel it.

### Recharge held constant in time, falsified against the precipitation record

The Kansas forward model carried one scalar recharge, uniform in space and constant
across all twenty-five years. Annual precipitation over these six counties runs from 291
to 674 mm, a factor of 2.31, so a recharge constant in time is falsified by the
precipitation record before any water-use report is opened. This is the same class of
defect as the saturated thickness: an independently published observation the model was
already wrong against.

The consequence is measurable and was measured before the fix was run. With recharge
constant, every year of head variation that weather caused is forced onto pumping. The
v3 closure's county-year error correlates with the precipitation anomaly at **+0.50**,
and the basin precipitation index explains only **7 per cent** of the closure's
interannual variance against **56 per cent** of the metered record's.

**Replaced by:** `ks_data.recharge_weight()`. One mean recharge rate is still estimated
and keeps its published prior; its time structure is each county's own precipitation
divided by that county's record mean, so the multiplier averages one by construction.
The parameter count is unchanged, which keeps the comparison against v3 clean. The
driver is NOAA nClimDiv county precipitation, which carries no water-use term.

**Prediction recorded before the rerun finished**, so that the result sizes it rather
than the mechanism: the weather share of the closure's interannual variance should rise
from 7 per cent towards the metered 56 per cent, the level error should fall, and the
recovery of the multi-year contrasts should survive. If the level error does not fall,
the mechanism was right about the model and wrong about what limits the score, and that
is what the entry will say.

### Scoring a water account on its level, replaced by scoring it on the change

The L2 rung scored every account on county-year level alone, on which a bar a reviewer
can build in three lines beats the closure: mapped irrigated area times one acre-foot per
acre scores 14.78 Mm3/yr, the same corrected by half the year's precipitation deficit
scores 11.48, and the closure scores 16.57.

**Not rejected, and published as it stands.** Both bars are meter-free, both are fair,
and the entry has to be the document that shows them. What the level metric hides is that
those bars are weather models. Against the basin precipitation index they carry 11 and 92
per cent of their own interannual variance from weather, against 56 per cent for the
metered record, and after weather is removed the metered record still falls by 40.9 Mm3
per standard-deviation year while the two bars fall by 2.8 and 4.0. They carry the half of
the signal that the weather causes and are blind to the half a policy changes.

**Added:** `scripts/18_ladder.py` publishes every meter-free bar on level and on change,
and `scripts/19_verify.py` scores the three multi-year contrasts fixed by the Kansas
policy record. The contrasts are the quantity a reduction target is written in.

### The aquifer as referee, on annual head anomalies. Rejected

`scripts/17_referee.py` asks whether the water levels alone can rank competing water
accounts with no meter present: each candidate account is held fixed and only the
aquifer's own properties are estimated against the head record, so what is reported is the
misfit the best admissible aquifer can reach. Seven accounts, two of them the flat account
rescaled by a factor the record cannot support, so the test could be seen to fail.

**Rejected on its first run.** The metered truth ranks last of seven and the flat account
inflated by half ranks first. The rank correlation between head misfit and metered error is
**-0.32**: the test anti-ranks.

**Why, and it is not the ensemble size.** Head misfit rank-correlates **+0.54** with an
account's interannual roughness, measured as the standard deviation of the year-over-year
change in the basin total. The observations are annual anomalies against each well's own
mean, so a smooth account is rewarded whatever its level, and the metered record is the
roughest account of the seven at 93.1 Mm3 against 5.7 for the flat one. The statistic
measures smoothness, not correctness.

**Not replaced in this pass.** The reformulation that follows from the diagnosis is to
score the account against the low-frequency component of the head record rather than the
annual anomaly, which is the timescale over which storage integrates. That is the same
conclusion the verification sweep reaches by a shorter route, and the sweep already carries
it: the closure recovers multi-year contrasts better than accounts that never see the
aquifer. The script and its negative result stay in the repository.

### Sign of the declared change, withdrawn as vacuous

The verification sweep reports that where the closure declares a change, the direction
agrees with the meters in 76 of 79 window pairs. That statistic was withdrawn before it
reached any judge-facing document.

**Rejected.** Abstraction fell over 119 of the 136 window pairs, so an estimator that
says "down" every time scores the same 76 of 79. Of the three declared pairs where the
meters show an increase, the closure gets none right. The statistic is satisfied by a
constant and measures nothing.

**Replaced by:** the area under the curve of the declaration against the metered
magnitude, **0.766** with a permutation p below 0.0001, and the correlation of the
estimated change with the metered change, r = 0.55 with an amplitude slope of 0.70. Those
are not satisfied by a constant. `scripts/19_verify.py` now records the negative result
alongside them, including what always saying "down" would score, so the vacuous statistic
cannot creep back.

### The precipitation forcing on recharge, run and rejected

The prediction recorded above was that giving recharge its observed precipitation forcing
would raise the weather share of the closure's interannual variance from 7 per cent
towards the metered 56 and lower the level error. **It did neither, and the configuration
is rejected.**

| | published, constant recharge | precipitation-forced recharge |
|---|---:|---:|
| level, Mm3/yr | **16.57** | 16.83 |
| relative error | **25.4%** | 28.0% |
| change over five-year periods, points | **8.7** | 9.1 |
| change over eight-year periods, points | **4.5** | 5.4 |
| crossover window | **4 years** | 5 years |
| weather share of its own variance | 7.2% | 5.8% |
| error against the precipitation anomaly | +0.50 | +0.48 |
| thickness multiplier on the published surface | 0.90 | 1.05 |
| head error after the two-stage budget | **2.94 m** | 6.12 m |

**The mechanism was right about the model and wrong about what limits the score.** The
thickness multiplier moves from 0.90 to 1.05, so the forced model agrees with the
published USGS surface more closely than the published configuration does. Everything
that is scored is worse.

**The diagnosis is in the error budget.** The head error the two-stage estimate returns
doubles, from 2.94 m to 6.12 m. The multiplier is applied per county, so recharge steps by
up to forty per cent across a county line that has no hydrogeological meaning, and the
head field cannot carry the discontinuity. The budget then inflates the head error to
cover it, the head leg is de-weighted, and the head leg is what carries the multi-year
change. The forcing has to be spatially smooth, interpolated from the county series or
taken from a gridded product, before this can be retried.

The code stays and is reachable under `KTAG=_v4`; the published rung stays `_v3`. This is
recorded rather than removed because the prediction was written down before the run
finished, and a prediction that fails is worth exactly as much as one that does not.
