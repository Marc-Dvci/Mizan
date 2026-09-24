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
that reason, and the rest of this entry is worth reading against four checkable facts
rather than against the ordering alone.

**The test used no pumping data.** `hp_satthk09` is a published grid of aquifer geometry
into which no water-use report enters. The prior ran from 20 m to 140 m and five of the
six counties sit at or below its lower bound, so the first configuration was not merely
missing the published value, it excluded it. That is a falsification against an
independent observation, and it was available before any meter was opened.

**The correction moved toward that observation, not toward the score.** The layer base is
now the published field times one estimated multiplier, so the parameter count is
unchanged and the comparison against the previous run is clean. The inversion chose 0.90,
within ten per cent of unity, which says the head record agrees with the published surface
rather than pulling away from it. A parameter changed to flatter a score does not land
there.

**It cost, and the cost is in the row above.** Relative error worsened from 23.3 to 25.4
per cent while mean absolute error improved from 17.65 to 16.57, Cheyenne moved the wrong
way, and no closure configuration beats mapped irrigated area times one acre-foot per acre
at 14.78 Mm³/yr. A change made to improve a score does not report that.

**What cannot be recovered is stated.** With six counties there is no untouched data left
to size a further effect on, so this rung cannot supply a clean second effect estimate for
the change itself. Both configurations remain in the repository and both remain runnable.
Nothing after this was tuned.

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

### The Kansas truth, labelled metered over the whole record. Corrected to the metered era

Reviewed 17 September 2026. The withheld series was read from the WIMAS history page
and described everywhere as per-water-right metered annual pumping, 2000 to 2024. The
page reports use; it does not say how the use was measured. WIMAS does, on every
report: the water-use file the site writes on request carries a code from the table in
KGS OFR 2005-30, where A, M and I are meter readings and G is hours of pump operation
times a rate. Read against that code, the meter-coded share of the block's reported
volume is 23.6 per cent in 2000, 28.7 in 2005, 52.1 in 2006, 68.7 in 2007, 86.4 in
2008, and 98.8 per cent or more in every year from 2009, which is the year GMD4 records
as the first with every well metered. So 2000 to 2008 was never metered truth, and the
label was wrong for nine of the twenty-five years.

The same file showed a second defect. The history page reports one point of diversion
at a time and the fetcher read one point per water right; use is filed per point, so a
right with several reporting points was under-counted. The complete per-point record
runs 6.9 to 9.8 per cent above the series first read in every year, and 2 to 16 per
cent by county.

**What was done.** `ks_fetch.fetch_county_use` retrieves the use file, `ks_data.reported_annual`
builds the truth from it with the code on each report, and `scripts/27_metered_era.py`
scores every published Kansas quantity again on 2009 to 2024, against the complete
record. Nothing on the estimate side was re-run: the posteriors are the published ones
and never saw a meter. A guard holds the era to the code shares, and its corruption,
the era started in 2008, fails it.

**What it did to the numbers.** The five-year change, the headline, moves from 8.7
points on 136 pairs to 9.4 on the 28 pairs the era admits; the best meter-free bar
moves from 9.9 to 12.3, so the margin widens from 1.2 to 2.9 points, the closure is
closer than the open loop on 75 per cent of pairs against 59, its interval covers 86 per
cent of the metered changes against 89, and the crossover stays at four years. The
interannual skill moves from +0.19 to +0.22. The level moves the other way: with the
truth 7 to 10 per cent higher the closure's basin bias goes from -7.4 to -15.5 per cent
and its relative error from 25.4 to 28.6, and every other account moves down with it.
Both named contrasts inside the era have the closure overstating the decline, -15.5
against a metered -6.0 on the GMD4 LEMA and -21.3 against -5.8 on the era's endpoints,
with the area-times-depth bar closer on both. The `_v3` results files are kept as they
were produced; `results/metered_era_v3.json` is the scoring to quote against meters.

### The published rung was not reproducible from the committed code

Found 18 September 2026 while preparing the next configuration. The commit that drove
recharge from the county precipitation record changed `ks_run.build` unconditionally,
and the commit that rejected that forcing added this log's entry and nothing else. So
`make kansas` on the published repository ran the rejected forcing under the `_v3` tag,
and the recharge guard required it to. The `_v3` results files predate the forcing
commit and are the numbers the submission quotes. **Fix:** the forcing is a driver flag,
`--forced-recharge`; `make kansas` runs without it and `make kansas-forced` with it, so
`_v3` and `_v4` are both reachable from one tree, and the guard now checks both
branches. No number moved.

### `_v5`: the storage coefficient and the conductivity prior from published maps, the deep-percolation share a parameter. Prediction written before the run

**Why.** On the metered era against the complete record, the closure is 15.5 per cent
low on the level, and the head leg alone is 50 per cent low in Cheyenne and 41 per cent
high in Sheridan. The `_v3` inversion holds one specific yield for the whole block and
settles it at 0.25 (heads only: 0.27) with recharge at 4.7 mm/yr, against a published
prior of 0.15 and 20 mm/yr. The USGS specific-yield map of the High Plains aquifer
(Cederstrand and Becker 1998, OFR 98-414) averages 0.17 over the block and runs from
0.13 in Decatur to 0.21 in Sherman. A storage coefficient fifty per cent above the
published map, with recharge at a quarter of its prior, is a water balance closing on
the wrong term: the model is short of a source of water and the head leg is buying it
with storage. The source the model lacks by construction is deep percolation under the
pivots, fixed at 30 per cent of the non-consumed fifth of pumping, about 6 per cent.

**What changes, all from published sources, no meter read.** Specific yield is the
USGS map times one multiplier held to about twenty per cent (log10 sd 0.08, bounds 0.6
to 1.6), as the layer base is already the USGS thickness grid times one multiplier. The
conductivity pilot-point prior is centred on the USGS conductivity map (OFR 98-548)
instead of a uniform 20 m/d. The deep-percolation share of pumping becomes a parameter
with a prior of 12 per cent, a factor of 1.6 either way, bounded at 6 to 24, which is
the range field studies under centre-pivot irrigation on the Kansas High Plains report.
Nothing else moves: same observations, same error budget, same ensemble size, same
localisation, recharge constant in time as in `_v3`.

**Prediction, to be scored against the metered era.** (1) The specific-yield multiplier
lands inside 0.8 to 1.25 and not on a bound. If it goes to 1.6 the model is still short
of water and the configuration is rejected, whatever the score. (2) The closure's basin
bias on the metered era moves toward zero from -15.5 per cent by at least five points,
and the head leg's county-level spread of bias shrinks from the present 91 points
(Cheyenne to Sheridan) to below 60. (3) The five-year change error stays within one
point of 9.4 or improves; a change of model structure that helps the level and costs
the change is recorded as that. (4) Recharge comes back toward its published prior,
above 10 mm/yr. Two of the four failing rejects the configuration; the published rung
stays `_v3` either way and both stay runnable.

### `_v5` scored against its own prediction. Two of four failed, so it is rejected

Run 18 September 2026, both stages, two independent prior ensembles (seeds 5 and 6),
scored on the metered era against the complete per-point record.

| | `_v3` seed 5 / 6 | `_v5` seed 5 / 6 |
|---|---|---|
| specific-yield multiplier | n/a | 1.157 / 1.142 |
| specific yield used | 0.249 / 0.251 | 0.201 / 0.199 |
| recharge, mm/yr | 4.7 / 6.2 | 11.0 / 8.0 |
| deep-percolation share | 0.06 fixed | 0.121 / 0.118 |
| basin bias, metered era | -15.5 / -14.7 % | -14.9 / -15.1 % |
| head-leg county spread of bias | 91 pts | 67 pts |
| level, relative error | 28.6 / 30.3 % | 26.0 / 28.1 % |
| **five-year change, metered era** | **9.35 / 8.39 pts** | **7.18 / 7.26 pts** |
| six-year change | 6.06 / 6.61 | 3.74 / 4.03 |
| interval coverage | 0.86 | 0.96 |

**The gate.** (1) The multiplier landed at 1.15, inside the band and off both bounds:
**pass**. (2) The basin bias had to improve by five points and the head leg's county
spread to fall below 60. They moved 0.6 points and to 67: **fail**. (3) The five-year
change had to stay within a point of 9.4: it improved to 7.2: **pass**. (4) Recharge
had to come back above 10 mm/yr: 11.0 on one seed and 8.0 on the other, mean 9.5:
**fail**. Two of four failed, so by the rule written before the run **the published rung
stays `_v3`**.

**Why that is the right call even though the scored number improved.** The reason given
for the change was a mechanism: the model is short of a source of water, the head leg is
buying it with storage, and supplying deep percolation from a published prior will move
the level bias. The level bias did not move. Adopting the configuration anyway would be
adopting a structural change on the strength of the metric it was supposed to be
incidental to, which is the failure mode `18_RESULT_SENSITIVE_CORRECTION_OPTIONS.md`
exists to prevent, and the meters had already been opened when the run finished.

**What it is evidence for, and what is reported.** The headline is not fragile. A
structural variant that changes the storage coefficient, the conductivity prior and the
recharge balance together moves the five-year change from 8.9 to 7.2 points averaged
over two seeds, against a `_v3` seed spread of 1.0, and never in the wrong direction.
Both configurations are runnable, `make kansas` and `make kansas-v5`, and both sets of
results files are kept.

### Why the twin's error rate does not transfer, measured rather than asserted

`scripts/28_net_inflow.py`, `make net-inflow`. The L0 twin abstracts 31.04 km3 and
depletes 29.06 km3 of storage, so **94 per cent of what it pumps comes out of storage**
and 6 per cent is replaced within the year. Northwest Kansas is not that basin. From
the WIMAS meters, the WIZARD winter levels and the USGS specific-yield map,

    N = Q + Sy * A * dh

gives net inflow of **152 Mm3/yr against 444 Mm3/yr pumped over 2009 to 2023, so the
aquifer replaces 34 per cent of its own abstraction and storage supplies 66**. The
county values run from 0.06 in Rawlins to 1.32 in Decatur, whose water table is flat.
This is the same quantity Butler et al. target for sustainability in these counties, and
it is computed here from the public record with no model in it.

**That is the size of the gap between the rungs.** The closure's information about
abstraction comes from storage. A basin where storage carries 94 per cent of pumping
presents the closure with a signal half again as large as one where it carries 66, which
is the direction the 5.2 against 25.4 per cent gap runs. It does not explain the whole
gap and is not claimed to. What it does is put a number on the statement that Kansas is
the harder of the two basins for this instrument, and the Saq, with recharge near zero,
sits at the twin's end of that range rather than at Kansas's.

### How large is the margin over the best meter-free bar, and how well is it resolved

Asked 18 September 2026, because the metered-era rescoring doubled the margin and a
number that moves that much under a change of scoring needs an error bar before it is
quoted. The margin is the mean absolute error of the best meter-free account minus the
closure's, over every non-overlapping window pair. Its uncertainty is taken by
leave-one-year-out: a year is dropped from the record, every window pair is rebuilt from
what remains, and the margin is recomputed, which is the jackknife over the only unit the
record has independent copies of.

| window | whole record, 2000-2024 | metered era, 2009-2024 |
|---:|---|---|
| 4 years | -0.16 ± 3.23 | +0.63 ± 5.25 |
| 5 years | **+1.57 ± 3.08** | **+2.94 ± 7.72** |
| 6 years | +3.19 ± 3.29 | +6.92 ± 11.97 |
| 7 years | +4.19 ± 4.11 | +5.99 ± 14.11 |
| 8 years | +4.63 ± 4.46 | 1 pair, not estimable |

**No window on either scoring puts the margin more than about one standard error above
zero.** On the metered era no window exceeds 0.6, and dropping a single year of the
sixteen takes the five-year margin from +2.94 to -1.10, so its sign is not robust to one
year. On the whole record at six years and longer the margin is about one standard error
and the worst single-year drop stays positive, +1.35 at six years, which is the strongest
form the comparison takes anywhere in this work.

**Why the pair count was never the sample size.** The submission already says the 136
pairs are a count of contrasts rather than independent cases. That statement was never
carried through to the margin itself. The metered era admits exactly **one** pair of
five-year windows that shares no year with another such pair, so 28 is a count of
overlapping contrasts drawn from sixteen years.

**What this does not change.** The coverage of the posterior interval, 86 per cent on the
era and 89 on the record against a nominal 90, is a calibration property of one account
and needs no comparison. That no meter-free bar states an interval at all is categorical.
The interannual anomaly skill, +0.22 leave-one-county-out against +0.24 for the oracle, is
estimated across counties rather than window pairs. Those three claims stand at the
precision they are stated.

**What it does change.** "The closure is out by 8.7 points against 9.9 for the best
meter-free bar" is a ranking of two point estimates whose difference this record cannot
resolve. It stays true as an arithmetic statement about the record and must not be read
as a demonstration that the closure is the better instrument at five years. The honest
form is the margin with its jackknife error beside it, and the longer windows, where the
effect is larger and survives dropping any single year, are where the comparison is
actually made.

### The headline error bar was computed on the wrong unit, and the claim shrinks

Found 18 September 2026, in the course of restating the headline. The gain over the
published open-loop account was first quoted as **20.7 ± 3.6 points, 5.7 standard
errors**, from a leave-one-year-out jackknife. That error bar is real but it answers
"how would this gain come out in another year, over these same six counties". It is not
the question the entry asks. The entry claims a method that transfers to a basin it has
not seen, and the unit that generalises is the **district**, of which this block has six.

| | gain | se by county (n=6) | se by year (n=16) | counties favouring the closure |
|---|---:|---:|---:|---:|
| published open loop | +20.7 | 12.1 (**1.7 se**) | 3.6 (5.7 se) | 4 of 6 |
| the same, efficiency fitted to the meters | +17.7 | 10.6 (**1.7 se**) | 4.3 (4.1 se) | 4 of 6 |
| mapped area x one acre-foot | -5.6 | 5.3 (-1.0 se) | 1.2 (-4.7 se) | 2 of 6 |
| the same, plus the precipitation deficit | -9.5 | 4.8 (-2.0 se) | 1.9 (-5.0 se) | 1 of 6 |

Per county the gain is Cheyenne -14.4, Rawlins +51.9, Decatur +60.4, Sherman +10.3,
Thomas +17.5, Sheridan -1.8. **It is carried by the two counties where the open-loop
account fails worst, at 77 and 81 per cent relative error, and it is negative in two of
the six.** So the honest figure is 1.7 standard errors, not 5.7, and the entry now says
so in the proposal, the technical note, the deck, `RESULTS.md` and the form answers.

A guard holds this: the county error must exceed the year error, the shipped prose must
carry the county figure, and the corruption, the naive independent error over the 96
county-years, must come out at less than half the county figure, which is what makes
quoting it an overclaim.

**Nothing about the point estimates changed.** 49.3 per cent against 28.6 stands, as does
every number in the level table. What changed is the strength claimed for the difference,
and it is smaller than the first reading said.

**The defensible form of the result.** Over this block the closure is substantially more
accurate than the technique in use, on meters, with the direction holding in four of six
counties; six counties of one climate cannot establish a seventh; and what licenses the
transfer argument is the mechanism and the ablation grid, not the width of this gap.

### Blind transfer to eighteen counties the closure has never seen. Predictions written before the meters were opened

**Why.** Every real-data error bar in the entry clusters by county, and the published
block has six. The gain over the open-loop account, 20.7 points, is 1.7 standard errors
on that unit; the change margin never clears one. Six counties of one climate cannot
establish a seventh, and no amount of rescoring the same six changes that. Every Kansas
input is a statewide or national public record keyed by county, and one block costs
about twenty minutes of solver time, so the constraint on the headline is bought with
compute and with a protocol that keeps the new counties blind.

**The blocks, chosen before any of their inputs was fetched.** Three contiguous groups
of six counties, each the size of the published block: `west` (Wallace, Logan, Gove,
Greeley, Wichita, Scott: GMD4 south and GMD1, the same aquifer, adjacent), `gmd3w`
(Hamilton, Kearny, Stanton, Grant, Morton, Stevens: GMD3 west, drier, deeper pumping,
larger declines) and `gmd3e` (Finney, Haskell, Gray, Seward, Meade, Ford: GMD3 east,
the heaviest abstraction in the state). Their county map comes from the Census county
polygons, because Finney and Ford are not rectangles; the published block keeps its tier
geometry and reproduces `_v3` exactly (the assembled observation vector, county map,
irrigated area and unmixing standard errors are byte-identical to the stored `_v3`
posterior inputs, checked before this entry was written).

**What is frozen.** The primary arm is `_v3` exactly as published: same prior, same
error budget procedure, same ensemble size, same localisation, same two-stage run. The
secondary arms are `_v5` (the map-based storage coefficient, rejected on the published
block by its own prediction) and the evapotranspiration leg alone. Nothing is tuned on
any new block, and nothing from the new blocks feeds back into the published one.

**How the blindness is held by the code.** `10_kansas_fetch.py --block <new>` fetches
the estimator's inputs only: diversion-point locations (licence data, used as spatial
weights), WIZARD levels, the rasters, county precipitation and the polygons. The WIMAS
use files are refused for a transfer block until the committed `DECISION_LOG.md`
carries the marker below for that block and the working copy of the log is clean, so
the meters cannot be opened in the same breath as the prediction is written. The runs
write their posteriors unscored (`11_kansas_run.py --block`), and `30_transfer.py`
scores them once the use files exist.

**Scoring rule, fixed now.** Each block is scored on its own metered era, found from the
WIMAS measurement codes by the rule of `27_metered_era.py`: the first year from which
every later year, block-wide and in every county, is at least 98 per cent meter-coded
by volume. The truth is the per-point use file. The accounts are the same five as the
headline: the closure, the evapotranspiration leg, area times one acre-foot per acre,
the same plus the precipitation deficit, and the open loop at 0.80. The headline
statistic is the gain over the open loop pooled over the eighteen new counties,
clustered by county, reported beside the published block and never mixed with it.

**Predictions, derived from the six-county results and scored as written.**

- P1. On the level, the closure has lower relative error than the published open loop in
  at least two thirds of the new counties (twelve of eighteen). On the published block
  it is four of six.
- P2. The 90 per cent interval covers the metered value in 0.80 to 0.97 of the new
  county-years. On the published block it is 0.91.
- P3. On the five-year change, the closure beats the open loop, mean over the new
  blocks. On the published block it is 9.4 against 12.3 points.
- P4. `_v5` beats `_v3` on the five-year change, mean over the new blocks. On the
  published block it did, 7.2 against 9.4, and was rejected on the level and recharge
  gates; this settles the change question on counties whose meters nobody has seen.
- P5. Area times one acre-foot per acre has a larger relative error in GMD3 than on the
  published block (23.0 per cent), because the applied depth in the southwest is not
  the northwest's. The rule of thumb is calibrated to a place; the closure reads the
  place.
- P6, stated as an expectation and scored: on the GMD3 blocks the closure's relative
  error on the level is below that of area times one acre-foot. On the published block
  the rule wins that comparison, 23.0 against 28.6, and the entry says so; if the
  southwest's depth runs at 1.4 acre-feet per acre or more the rule's bias alone
  exceeds the closure's whole error.

Whatever comes out is reported. A failed prediction is recorded as failed, with its
number, in the same table as the passes.

TRANSFER PREDICTIONS COMMITTED FOR BLOCK west
TRANSFER PREDICTIONS COMMITTED FOR BLOCK gmd3w
TRANSFER PREDICTIONS COMMITTED FOR BLOCK gmd3e

### The blind transfer, scored. Five of six predictions held, and the headline moves from six counties to twenty-four

Run 18 September 2026. The three blocks were fetched without their use files, run under
the frozen `_v3` and the `_v5` arm, the posteriors committed unscored (`62a7044`,
`c3c6373`, `6ae2d8e`), and only then were the meters opened, each block through the
fetcher's own refusal gate.

**The result.** Over the eighteen counties the method has never seen, the closure removes
**10.9 ± 5.6 points** of relative error from the published open-loop account, clustered by
county, which is **2.0 standard errors** and positive in **13 of 18**. Pooled over all
twenty-four counties it is **13.4 ± 5.1, 2.6 standard errors, 17 of 24**. The published
block alone gave 20.7 ± 12.1, 1.7 standard errors on six counties. The point estimate is
smaller off the block it was built on and the claim is better resolved, which is the
trade the experiment was run to make.

| block | metered era | county-years | closure | ET leg | area x 1 af | + weather | open loop | gain vs open loop |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GMD4 north, the published block | 2009-2024 | 96 | 28.6% | 20.4% | 23.0% | 19.1% | 49.3% | +20.7 ± 12.1 |
| West Kansas, GMD4 south and GMD1 | 2014-2024 | 66 | 37.3% | 41.5% | 34.9% | 29.3% | 44.9% | +7.6 ± 14.6 |
| GMD3 west | 2015-2024 | 60 | 24.3% | 22.4% | 32.8% | 38.5% | 38.8% | +14.5 ± 7.5 |
| GMD3 east | 2009-2024 | 96 | 21.2% | 19.7% | 27.2% | 29.6% | 31.9% | +10.7 ± 6.7 |

**The predictions, as written.**

| | prediction | result | |
|---|---|---|---|
| P1 | closure beats the open loop on the level in at least 12 of the 18 new counties | 13 of 18 | held |
| P2 | 90 per cent interval covers 0.80 to 0.97 of new county-years | 0.982 | **failed** |
| P3 | closure beats the open loop on the five-year change, mean over new blocks | 7.45 vs 7.55 points | held |
| P4 | `_v5` beats `_v3` on the five-year change, mean over new blocks | 7.09 vs 7.45 | held |
| P5 | area x one acre-foot has a larger error in GMD3 than on the published block | 30.0% vs 23.0%; GMD3 meters at 1.48 acre-feet per acre | held |
| P6 | on the GMD3 blocks the closure beats area x one acre-foot on the level | 22.7% vs 30.0% | held |

**P2 failed in the conservative direction and that is the finding, not an excuse.** The
interval is too wide off the block it was calibrated on: 0.98 of new county-years fall
inside a nominal 90 per cent band, against 0.91 on the published block, and the same
excess appears at the 50 and 80 per cent levels (0.61 and 0.92 on GMD3 east against 0.50
and 0.77 at home). A posterior that over-covers overstates its own uncertainty, which is
the safe direction for a regulator and a real defect in the calibration. The two-stage
error budget reads its structural term from the block's own residual, and on a block with
sparser levels and larger county volumes that term comes out large; the entry now says
the interval is conservative off the tuned block rather than calibrated everywhere.

**What P5 and P6 settle.** The two arithmetic bars are calibrated to a place. Northwest
Kansas irrigates at 1.09 acre-feet per acre, which is why one acre-foot per acre beats the
closure there; GMD3 meters 1.42 to 1.53, and the same rule loses to the closure by 7.3
points on those blocks. The rule does not transport and the closure reads the place, which
is the argument the entry makes for an unmetered basin and now has a measurement behind
it rather than an assertion.

**What did not transfer.** The west block is the weakest of the four: the closure is out by
37.3 per cent there, worse than both arithmetic bars, and Logan county alone is -55 points
against the open loop. It is the thinnest saturated thickness of the four (the published
USGS surface has a median under seven metres over that block, against 34 in GMD3 east),
only 187 wells carry a usable record, and the head leg has the least to work with. It is
reported at the same size as the blocks that went the entry's way.

**The change comparison is still the weaker one.** Pooled over the new blocks the closure
is 7.45 points against 7.55 for the open loop, a margin of 0.1 points, and it is behind
the open loop on GMD3 east (6.8 against 5.4). Against the two arithmetic bars it is ahead
on every block. The level comparison remains the one the record resolves.

**Nothing on the published block changed.** No parameter, prior or procedure was touched
for the transfer; the published block reproduces `_v3` byte for byte under the block table
and a guard holds that.

### The failed prediction, measured and corrected out of sample

Run 18 September 2026, after the transfer, `make interval` (`scripts/31_interval.py`).
P2 was the one prediction the transfer failed: the 90 per cent interval covered 0.98 of
new county-years against a band of 0.80 to 0.97. This asks what the defect is and what a
correction would be worth, and it answers both without touching a posterior.

**What it is.** The factor on the posterior spread, about its own mean and in the log the
inversion parameterises, that makes the 90 per cent interval nominal:

| block | factor | cover 50 | cover 80 | cover 90 |
|---|---:|---:|---:|---:|
| GMD4 north, where the error budget was estimated | 1.00 | 0.50 | 0.77 | 0.91 |
| West Kansas | 0.65 | 0.71 | 0.95 | 0.98 |
| GMD3 west | 0.84 | 0.52 | 0.85 | 0.97 |
| GMD3 east | 0.80 | 0.61 | 0.92 | 0.99 |

The block the two-stage error budget was estimated on wants no correction at all. All
three blocks it was not estimated on want the same one, 0.65 to 0.84, mean 0.82, spread
0.14. **The defect is not a property of any block; it is the error budget being read in
sample.** The structural term comes from each block's own converged residual, and where
the water-level network is sparser and the county volumes larger that term comes out too
large for the fit it is describing.

**What a correction buys, scored where it was not fitted.** Leave-one-block-out: the
factor applied to a block is the mean of the factors fitted on the other blocks, so no
block enters its own calibration. Pooled over the 222 county-years of the blocks the
method had never seen, coverage goes **0.98 to 0.92 at the 90 per cent level, 0.91 to
0.81 at 80, and 0.62 to 0.50 at 50**: nominal at all three, out of sample. It costs CRPS,
26.76 to 27.16 Mm3/yr, 1.5 per cent, which is what a narrower interval costs when the
point estimate carries bias, and that trade is reported rather than hidden.

**What is not claimed.** The rule fitted on the published block alone, which is all the
entry had before the transfer, is a factor of 1.00 and corrects nothing: 0.98, 0.97 and
0.99 on the three new blocks. **The correction is only visible because the blocks exist**,
which is the second thing the transfer bought. Fitting the factor on the block it is
applied to reaches nominal coverage by construction and is an oracle; a guard holds the
leave-one-out structure and rejects that form.

**The shipped scores stay uncalibrated.** No posterior is rewritten and no shipped number
moves; the guard asserts the script writes no npz. The documents now state the interval
as conservative away from the tuned block, with the size of the correction a next basin
would need and its out-of-sample score.

### Is the deformation leg observable over the Saq's own pivots? Measured, from public data

Run 18 September 2026, `make saq-insar` (`scripts/32_saq_insar.py`). Two of the four legs
are exercised on real data and the deformation leg is the one the entry has never
observed on a real aquifer. The question a juror asks about the Saq is not whether
interferometry works but whether it works over irrigated ground in Al Jawf, where active
centre pivots decorrelate the radar.

**The route.** COMET's LiCSAR service publishes processed Sentinel-1 interferograms, and
frame `116A_05991_141313` (track 116, ascending) contains the Wadi As-Sirhan pivot field.
Thirty of its thirty-eight published interferograms carry usable phase over the Al Jawf
box, eleven epochs from 30 December 2022 to 11 May 2023. The pivot mask is the ESA
WorldCover 2021 cropland class averaged onto the 111 m radar grid, so the classification
carries no radar information and the contrast below is not circular. No account, no
credential, two public hosts.

**The contrast, over 2.6 million pixels.**

| | pivot fields | desert |
|---|---:|---:|
| mean coherence over 30 interferograms | 0.14 | 0.52 |
| share of pixels above 0.3 | 0.12 | 0.82 |
| share above 0.5 | 0.02 | 0.62 |

The pivots decorrelate exactly as expected and the ground between them holds phase in
four pixels out of five. **The deformation leg is readable over this basin, on the
interpivot ground, which is where a compaction signal from the aquifer would be read
anyway.** The coherence map reproduces the pivot pattern of an independent land-cover
map, which is its own check that the two are seeing the same ground.

**What this published window does not do.** Eleven epochs over 0.36 years: the
line-of-sight rate comes out -7.2 ± 35.0 mm/yr over the pivots and -26.1 ± 42.2 over an
independent desert control levelled against the far field. At two standard errors that
span resolves 70 mm/yr, against a published Saq subsidence range of 4 to 15. **This
frame's published archive establishes where the phase can be read, not the rate.** A rate
needs the full Sentinel-1 archive over that frame, which is ASF HyP3 plus a time-series
inversion and an Earthdata login; the entry states the observability result and the route
to the rate, and claims no rate.

### The Saq pilot, pre-registered before any Saudi meter record is seen

Written 24 September 2026, before the pilot exists and before any MEWA meter record,
Saq monitoring-well head or Saq aquifer model has been seen by this project. The Kansas
blind transfer is the template: the scoring rule and the predictions go on public record
first, and whatever comes out is reported, a failed prediction in the same table as the
passes.

**The unit and the truth.** The account is the Wadi As-Sirhan pivot block, the 2,541 km²
delineated for 2015 by `20_aljawf.py`, inside Sentinel-1 frame 116A. The truth is the MEWA
agricultural well-meter record over that block. A share of the metered wells, drawn at
random by district before any record is read, is withheld from the estimator; the account
is scored on the fields those withheld wells serve, aggregated to the smallest unit the
meter records support. The basin total is reported over a footprint of whole gravimetric
mascons, where the averaging gain is one by construction, as the calibration rule already
states.

**The configuration is frozen before the withheld meters are opened.** The conceptual
model, priors and error budget are fixed and committed first. The interval carries the
spread factor Kansas measured off the block it was fitted on, the mean over the four
Kansas blocks, 0.8225 (`31_interval.py`), applied from the start and never refitted on
the Saq.

**The accounts compared.** The closure; the evapotranspiration leg alone; the published
open-loop method, each evapotranspiration product over a fixed 0.80; and the published
5.5 bcm/yr for 2015 (López Valencia et al. 2020).

**Predictions, derived from the Kansas results and scored as written.**

- S1. The efficiency the closure recovers over the block, consumptive use over
  abstraction, posterior median, is below 0.80. In Kansas the efficiency fitted to each
  block's own meters was 0.63, 0.78, 0.51 and 0.62, below 0.80 in all four blocks, and
  MEWA states national irrigation efficiency at around 50 per cent.
- S2. On the level, the closure has lower relative error than the open loop at 0.80 in at
  least two thirds of the scoring units. In Kansas it was 13 of 18 counties never seen.
- S3. The 90 per cent interval, with the Kansas spread factor applied, covers the withheld
  meters in 0.80 to 0.97 of scoring unit-years. This is Kansas P2, which failed by
  over-covering, restated with the correction that failure measured.
- S4. Over the coherent desert ground inside the pivot field's outline (mean coherence at
  least 0.3 in the 30-interferogram probe), the full Sentinel-1 stack over frame 116A
  gives a vertical rate below zero, subsidence, resolved from zero at two standard errors.
  Published rates for Saudi agricultural areas are 4 to 15 mm a year (Othman et al. 2018)
  and 1.1 to 5.1 in a separate estimate.

SAQ PREDICTIONS COMMITTED
