# Results

Generated into `RESULTS.md` by `make report`, which runs
`scripts/07_report.py` over the results files in this repository. Every
number in the proposal, the deck, the film and the technical note comes from
here and from nowhere else.


## The test basin

| Quantity | Value |
|---|---|
| Domain | 100 x 100 km, two layers, 20 years, 240 monthly stress periods |
| Truth grid / estimator grid | 1 km / 2 km |
| Districts | 9 |
| Total abstraction | 31.04 km3 (1552 Mm3/yr) |
| Storage depleted | 29.06 km3 |
| Peak head decline | 55.8 m |
| Peak subsidence | 84.5 cm (4.22 cm/yr) |
| **Storage capacity destroyed permanently** | **2.768 km3, 9.5% of simulated storage depletion** |
| Irreversible share of peak subsidence | 96% |
| Observations | et 2160, grace 240, insar 5120, head 8130 |

## Ablation grid: what each observation is worth

Scored on district-annual abstraction against withheld truth. MAE in Mm3/yr.

| Observations | MAE | MAPE | bias | 50% | 80% | 90% | interval width |
|---|---:|---:|---:|---:|---:|---:|---:|
| open loop, efficiency fixed at 0.80 | 16.66 | 13.6% | -13.57 | none | none | none | none |
| open loop, efficiency fitted to truth at 0.756 | 14.33 | 11.2% | -4.23 | none | none | none | none |
| prior, no data | 170.47 | 184.6% | +160.61 | 48% | 60% | 71% | 717.7 |
| evapotranspiration only | 18.47 | 14.3% | -0.45 | 47% | 68% | 84% | 97.4 |
| heads only | 90.99 | 51.6% | +90.97 | 3% | 12% | 32% | 158.3 |
| gravity only | 93.48 | 82.2% | +6.68 | 35% | 78% | 93% | 359.1 |
| deformation only | 146.35 | 88.8% | +145.51 | 13% | 28% | 43% | 299.2 |
| evapotranspiration + heads | 6.76 | 5.2% | -0.64 | 76% | 99% | 100% | 77.4 |
| evapotranspiration + gravity | 15.59 | 12.8% | -2.25 | 53% | 69% | 84% | 93.7 |
| evapotranspiration + deformation | 11.35 | 7.9% | -4.00 | 72% | 89% | 92% | 83.9 |
| heads + gravity + deformation | 8.46 | 6.7% | +0.22 | 87% | 97% | 98% | 83.1 |
| **satellites only, no wells at all** | 9.85 | 7.4% | -2.73 | 71% | 86% | 94% | 82.3 |
| all four, 10 wells instead of 97 | 9.18 | 7.0% | -4.06 | 67% | 90% | 98% | 79.0 |
| **all four, coupled closure** | 6.72 | 5.4% | -2.32 | 77% | 98% | 99% | 70.3 |
| meters on every district, no satellites | 12.26 | 7.1% | -4.15 | 88% | 100% | 100% | 121.1 |
| all four, plus one metered district | 6.96 | 4.5% | +2.42 | 86% | 99% | 100% | 72.1 |
| all four, plus three metered districts | 7.81 | 5.4% | -5.97 | 78% | 94% | 99% | 68.1 |

The closure reduces the error of the published open-loop account by a factor of **2.5**, and of the same form with its efficiency fitted against the answer by a factor of **2.1**.

## What the observation set could not resolve

| Observations | directions resolved to 90% | unresolved | widened | directions constrained |
|---|---:|---:|---:|---:|
| **null: two independent prior ensembles, no data at all** | **19** | **94** | **89** | **58.3** |
| evapotranspiration only | 83 | 32 | 31 | 121.9 |
| heads only | 92 | 34 | 31 | 125.1 |
| gravity only | 9 | 113 | 101 | 37.1 |
| deformation only | 54 | 69 | 66 | 86.9 |
| evapotranspiration + heads | 99 | 22 | 20 | 135.1 |
| evapotranspiration + gravity | 82 | 33 | 31 | 121.6 |
| evapotranspiration + deformation | 86 | 30 | 27 | 125.3 |
| heads + gravity + deformation | 90 | 34 | 32 | 123.4 |
| **satellites only, no wells at all** | 88 | 29 | 27 | 126.0 |
| all four, 10 wells instead of 97 | 89 | 27 | 26 | 127.8 |
| **all four, coupled closure** | 100 | 22 | 20 | 135.4 |
| meters on every district, no satellites | 57 | 57 | 55 | 95.6 |
| all four, plus one metered district | 100 | 22 | 19 | 135.5 |
| all four, plus three metered districts | 98 | 22 | 19 | 136.0 |

Out of 180 directions of the district-year abstraction vector. A direction whose posterior variance exceeds its prior variance has learned nothing and is counted as widened rather than as negative information.

## Recovery of the quantities the open-loop method assumes

| Observations | consumptive fraction MAE | pre-canopy share MAE | preconsolidation offset |
|---|---:|---:|---:|
| evapotranspiration only | 0.089 | 0.057 | 22.34 m against 12.00 m |
| heads only | 0.062 | 0.017 | 11.06 m against 12.00 m |
| gravity only | 0.054 | 0.051 | 18.26 m against 12.00 m |
| deformation only | 0.053 | 0.059 | 22.62 m against 12.00 m |
| evapotranspiration + heads | 0.037 | 0.018 | 11.47 m against 12.00 m |
| evapotranspiration + gravity | 0.078 | 0.053 | 19.44 m against 12.00 m |
| evapotranspiration + deformation | 0.047 | 0.047 | 20.20 m against 12.00 m |
| heads + gravity + deformation | 0.057 | 0.032 | 11.99 m against 12.00 m |
| **satellites only, no wells at all** | 0.040 | 0.049 | 10.10 m against 12.00 m |
| all four, 10 wells instead of 97 | 0.040 | 0.042 | 12.25 m against 12.00 m |
| **all four, coupled closure** | 0.033 | 0.030 | 11.92 m against 12.00 m |
| meters on every district, no satellites | 0.061 | 0.052 | 22.34 m against 12.00 m |
| all four, plus one metered district | 0.036 | 0.019 | 11.93 m against 12.00 m |
| all four, plus three metered districts | 0.024 | 0.025 | 12.10 m against 12.00 m |

## Error budget

Observations are weighted at instrument error. The structural error of the coarse
forward model is estimated after the fact from the full-information residual.

| Leg | instrument | residual | structural | lag-1 | independence inflation |
|---|---:|---:|---:|---:|---:|
| et | 1.84e+06 | 1.857e+06 | 2.482e+05 | 0.01 | 1.01 |
| grace | 14 | 17.96 | 11.26 | 0.03 | 1.03 |
| insar | 0.006 | 0.009759 | 0.007697 | 0.37 | 1.48 |
| head | 0.5 | 0.6868 | 0.4708 | 0.26 | 1.31 |

On every leg the structural component comes out below the instrument error, so no inflation is applied.

## Convergence

| iteration | MAE | bias | 90% coverage | interferometry | heads | gravity |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 170.47 | +160.61 | 71% | 55.12 mm | 19.970 m | 6991.1 mm |
| 1 | 75.24 | +73.34 | 94% | 47.04 mm | 3.351 m | 1025.9 mm |
| 2 | 53.93 | +53.85 | 97% | 14.59 mm | 2.041 m | 554.0 mm |
| 3 | 15.81 | +13.12 | 99% | 14.85 mm | 0.804 m | 123.6 mm |
| 4 | 9.66 | -2.42 | 100% | 11.04 mm | 0.773 m | 16.1 mm |
| 5 | 8.87 | -2.85 | 100% | 9.30 mm | 0.755 m | 15.4 mm |
| 6 | 8.05 | -2.56 | 100% | 8.57 mm | 0.696 m | 14.9 mm |
| 7 | 7.22 | -2.68 | 99% | 8.31 mm | 0.686 m | 14.9 mm |
| 8 | 6.61 | -0.82 | 99% | 8.52 mm | 0.687 m | 17.0 mm |
| 9 | 7.37 | +1.76 | 99% | 8.28 mm | 0.666 m | 24.9 mm |

## Decision layer

Zones: 45 drawdown strata. Time invariance of the response matrix: mean worst-case error 0.8% of the pulse response.

Permanent loss already taken over the record: 3037 Mm3.

**How much can be taken.** Permanent storage loss over the next twenty years, in full MODFLOW across the posterior.

| delivered, km3 | cut | mean loss, Mm3 | 10th | 90th |
|---:|---:|---:|---:|---:|
| 27.96 | 0% | 4075 | 3846 | 4332 |
| 25.17 | 10% | 3663 | 3457 | 3894 |
| 22.37 | 20% | 3252 | 3067 | 3457 |
| 19.57 | 30% | 2840 | 2678 | 3020 |
| 16.78 | 40% | 2428 | 2289 | 2583 |
| 13.98 | 50% | 2017 | 1901 | 2146 |

Cutting delivery by 13.98 km3 over the horizon avoids 2058 Mm3 of permanent capacity loss: **147 Mm3 of capacity per km3 not taken.**

Every frontier value above is evaluated directly in full MODFLOW across the posterior. Experimental spatial-allocation diagnostics are retained in `results/allocation.json` and are not submission results.

## Value of information


**Forecast: basin abstraction over the last five years.** Prior standard deviation 8.774e+08, posterior 8.92e+07.

| rank | instrument | uncertainty removed |
|---:|---|---:|
| 1 | meter_d2 | 1.451e+07 |
| 2 | meter_d0 | 1.43e+07 |
| 3 | meter_d7 | 1.413e+07 |
| 4 | meter_d4 | 1.399e+07 |
| 5 | meter_d6 | 1.358e+07 |
| 6 | meter_d5 | 1.333e+07 |
| 7 | meter_d1 | 1.261e+07 |
| 8 | meter_d3 | 1.145e+07 |

Forward selection of 20 instruments takes the posterior standard deviation from 8.92e+07 to 4.08e+07, a **54% reduction**.

**Forecast: storage capacity destroyed permanently.** Prior standard deviation 2.612e+09, posterior 3.346e+07.

| rank | instrument | uncertainty removed |
|---:|---|---:|
| 1 | gnss_0 | 4.362e+05 |
| 2 | gnss_18 | 3.536e+05 |
| 3 | gnss_10 | 3.087e+05 |
| 4 | gnss_57 | 1.852e+05 |
| 5 | gnss_11 | 1.839e+05 |
| 6 | piezo_0 | 1.683e+05 |
| 7 | piezo_9 | 1.668e+05 |
| 8 | piezo_37 | 1.664e+05 |

Forward selection of 20 instruments takes the posterior standard deviation from 3.346e+07 to 3.168e+07, a **5% reduction**.

Schur complement cross-checked against pyEMU: fc_q_last5 relative difference 5.5e-14, fc_perm_loss relative difference 1.6e-11.

## Detecting abstraction no evapotranspiration product can see

A withdrawal of 40 Mm3/yr with no canopy, and therefore no evapotranspiration signature, was planted in district D3: 39% of that district's abstraction.

- The open-loop account attributes **44%** of that district's true abstraction. It cannot see the withdrawal at all.
- The closure attributes **84%**.
- On the ratio of closure estimate to what consumptive use can explain, the district sits **11.4 standard deviations** above every other district.
- Basin-wide error: closure 10.04 Mm3/yr against 21.11 for the open-loop account.

## Robustness

The rows that carry the argument, repeated on independent prior ensembles and independent ES-MDA perturbations. One seed is not evidence; the spread is what the difference between two rows has to beat.

| observations | seed 5 | seed 6 | seed 7 | mean | spread |
|---|---:|---:|---:|---:|---:|
| **all four, coupled closure** | 6.72 | 6.27 | 7.27 | **6.75** | 1.00 |
| heads + gravity + deformation | 8.46 | 8.07 | 10.11 | **8.88** | 2.04 |
| **satellites only, no wells at all** | 9.85 | 10.65 | 10.01 | **10.17** | 0.80 |
| evapotranspiration only | 18.47 | 18.49 | 17.54 | **18.17** | 0.95 |

Differences smaller than the spread in this table are not claimed.

**A truth with no district spread in the consumptive fraction.** Every district given the same fraction, 0.79, which is the case most favourable to the open-loop form.

| truth | open loop at 0.80 | open loop, constant fitted | closure |
|---|---:|---:|---:|
| districts differ, 0.69 to 0.88 | 16.66 | 14.33 | **6.72** |
| every district identical | 19.75 | 7.52 | **5.38** |

## L2 Kansas: against real metered abstraction

Six counties of the Northwest Kansas groundwater management district over the Ogallala, 2000 to 2024, 1,355 km2 irrigated. The scored quantity is county-annual abstraction against **per-water-right metered pumping** published by the Kansas Department of Agriculture: 3,545 water rights, none of them unreadable.

The estimator sees 7,637 well-year head anomalies from 358 wells and 150 county-year evapotranspiration volumes. It never sees the metered volumes: the licensed points of diversion enter as spatial weights, and the use filed against them is the withheld truth.

| observations available to the estimator | MAE, Mm3/yr | MAPE | 90% coverage |
|---|---:|---:|---:|
| mapped irrigated area times one acre-foot per acre, no data at all | 14.78 | 21.7% | none produced |
| the same prior drawn as an ensemble, no data at all | 16.44 | 26.6% | none produced |
| open loop, efficiency fixed at 0.80 | 22.52 | 42.2% | none produced |
| open loop, efficiency fitted to the meters at 0.694 | 20.51 | 40.6% | none produced |
| evapotranspiration only | 13.88 | 18.8% | 100% |
| heads only | 18.45 | 29.1% | 91% |
| **evapotranspiration + heads, closure** | **16.57** | **25.4%** | 95% |

The row above is the per-site error budget. Under the pooled budget on the same published thickness the closure scores 16.92 Mm3/yr at 26.5% with 97% coverage, against 16.57 at 25.4% with 95%. Both are reported and neither was selected against the meters.

**The layer thickness is an observation, not a parameter.** The first Kansas configuration estimated one global saturated thickness and settled at 79 m. The USGS High Plains saturated-thickness grid, sampled onto the same model grid, gives a block mean of 20.4 m and county means of 19.7, 16.6, 13.2, 29.4, 23.6 and 19.1 m. The prior did not merely miss that: it ran from 20 m to 140 m, so five of the six counties sat at or below its lower bound. The surface that falsified it is an independent published observation into which no water-use report enters, so the test itself needed no meter; the decision log records that the change was made after the first Kansas score was known. The layer base is now that field times one estimated multiplier, so the parameter count is unchanged.

Posterior nuisances: a multiplier of 0.90 on the published saturated thickness, which puts the layer at 18.4 m over the block, recharge 5 mm/yr, consumptive fraction by county CN 0.80, RA 0.72, DC 0.82, SH 0.76, TH 0.68, SD 0.67. The multiplier lands within ten per cent of unity, so the head record is consistent with the published surface rather than pulling away from it.

**One declaration on the specific yield.** The posterior specific yield is 0.249, which sits above the range published for the Kansas High Plains, and it has to be read as an upper bound rather than as a retrieval. Putting the true thickness in costs part of the prior ensemble, because thin low-storage members dewater and fail to converge. Measured on 80 members, 27 of which failed, the failed members sit 0.07 standard deviations from the converged ones on block abstraction, so the scored quantity is unaffected, but 0.77 on specific yield and 0.46 on the thickness multiplier. The truncation is selective in exactly those two directions and the number is reported with that attached.

**The level is the easy part.** Irrigated area times one published applied depth already lands close on a county mean, so the county-year anomaly about each county's own record mean is what an estimator has to earn. The signal is 12.01 Mm3/yr mean absolute, and a flat-in-time estimate scores exactly that.

| estimate | MAE | anomaly MAE | anomaly skill |
|---|---:|---:|---:|
| each county's own 25-year mean, flat in time | 12.01 | 12.01 | +0.00 |
| open loop at 0.80 | 22.52 | 14.11 | -0.17 |
| heads only | 18.45 | 12.32 | -0.03 |
| evapotranspiration only | 13.88 | 10.83 | +0.10 |
| evapotranspiration + heads, closure | 16.57 | 10.27 | +0.15 |

**What the two legs could resolve**, on the same statistic L0 reports, against the null of two independent prior ensembles with no data assimilated.

| observations | resolved to 90% | unresolved | widened | directions constrained |
|---|---:|---:|---:|---:|
| **null: no data at all** | **9** | **80** | **76** | **44.8** |
| evapotranspiration only | 7 | 83 | 80 | 40.9 |
| heads only | 45 | 52 | 49 | 76.5 |
| evapotranspiration + heads, closure | 39 | 54 | 51 | 73.6 |

Out of 150 county-year directions. Read against the null row, not against zero.

Above the null the head leg carries Kansas at +31.6 directions and the closure sits at +28.8, while the evapotranspiration leg alone at -3.9 is below what nothing does: a 1 km product over a block that is 14 per cent irrigated carries a structural error large enough to cancel the leg. Adding it to the heads costs 2.8 directions of resolution and buys the level and the interannual amplitude reported below, which is a trade the two tables have to be read together to see. The same statistic at L0 put the four-leg closure +77.1 above its own null out of 180 directions. The real two-leg configuration extracts about a third as much information per direction as the synthetic one did, and it said so before the meters were opened.

**The interannual amplitude, calibrated without an oracle.** The estimate carries real year-to-year information: its county-year anomaly correlates with the metered one. With a correlation below one, the amplitude that minimises mean absolute error is smaller than the estimate's own, and one scalar per county supplies it. Fitted against the meters that scalar is an oracle, so it is fitted leave-one-county-out: every county's factor comes from the other five and no county enters its own fit.

| estimate | correlation | amplitude ratio | raw skill | leave-one-county-out | oracle |
|---|---:|---:|---:|---:|---:|
| open loop at 0.80 | 0.64 | 1.47 | -0.17 | +0.13 | +0.16 |
| heads only | 0.38 | 0.79 | -0.03 | +0.03 | +0.06 |
| evapotranspiration only | 0.65 | 0.97 | +0.10 | +0.17 | +0.18 |
| **evapotranspiration + heads, closure** | 0.64 | 0.84 | +0.15 | **+0.18** | +0.18 |

The factor is 0.68 to 0.75 across the six folds, so holding a county out costs nothing against the oracle. This is the operational requirement the value-of-information layer reached from the other direction: a few metered counties calibrate the amplitude for the rest.

## What a reduction target actually asks, and who can answer it

A water account is usually judged on its level. A regulator with a reduction target is not asking for a level. Saudi Arabia has published a 90 per cent reduction target for non-renewable groundwater, and Kansas writes its Local Enhanced Management Areas as a percentage cut against a stated baseline period. Both are questions about a change between two multi-year periods, with an interval on it.

Every meter-free account that can be written down from the same public data, scored against the same withheld meters, on both quantities.

| account | level, Mm3/yr | change over 5-year periods, points | weather share of its own variance |
|---|---:|---:|---:|
| mapped irrigated area x one acre-foot per acre | 14.78 | 15.0 | 11% |
| the same, plus half the year's precipitation deficit | 11.48 | 11.4 | 92% |
| unmixed evapotranspiration over a fixed efficiency of 0.80 | 22.52 | 9.9 | 25% |
| **the closure, evapotranspiration and heads** | 16.57 | **8.7** | 7% |

The metered record itself carries 56 per cent of its interannual variance from precipitation, and after weather is removed it still falls by 40.9 Mm3 per standard-deviation year. The two arithmetic bars are 11 and 92 per cent weather and keep 2.8 and 4.0 of that trend. They carry the half of the signal the weather causes and they are blind to the half a policy changes.

The change is scored over every pair of non-overlapping 5-year windows the record admits, 136 of them, rather than over a chosen contrast. The closure's 90 per cent interval contains the metered change in 89 per cent of pairs. Where it declares a change the metered change averages 19.7 per cent against 11.2 per cent where it declares none, an area under the curve of 0.766 with a permutation p below 0.0001.

**The direction of the change is not a test on this record and is not reported as one.** Abstraction fell over 119 of the 136 pairs, so an estimator that says down every time scores 96 per cent, which is what the closure scores. The magnitude above is the test.

The length of the window is not a free choice. A weather model carries the high-frequency half of the signal and saturates; the aquifer integrates storage and keeps improving.

| averaging window | pairs | closure | best meter-free bar | closure interval covers |
|---:|---:|---:|---:|---:|
| 2 years | 253 | 14.0 | 12.1 | 88% |
| 3 years | 210 | 13.0 | 11.3 | 81% |
| 4 years | 171 | **10.9** | 11.0 | 84% |
| 5 years | 136 | **8.7** | 9.9 | 89% |
| 6 years | 105 | **6.4** | 9.2 | 95% |
| 7 years | 78 | **5.2** | 9.1 | 97% |
| 8 years | 55 | **4.5** | 9.1 | 100% |

**The closure beats every meter-free bar from a 4-year window upward, and the gap widens with every year added.** Below that the best of them is a weather model and it is the better instrument. That crossover is a design rule for a monitoring programme and it is measured against real meters.

The posterior spread on a basin-wide five-year-against-five-year contrast is 9.4 percentage points, so this observing system separates a real reduction from no change at 12.1 per cent, at 90 per cent one-sided confidence.

**Where it runs out.** The Sheridan-6 Local Enhanced Management Area covers 256 km2 inside a 2,331 km2 county. Against the four clean neighbouring counties the meters give a difference in differences of -8.7 points; the closure gives +13.4 plus or minus 18.0, with the wrong sign and a 90 per cent interval of [-15.1, +41.9]. A policy on a tenth of a county is below what this observing system resolves, and the resolution analysis said so before the meters were opened.


## L3 Al Jawf: how far apart the published instruments are on the Saq

No metered abstraction exists for this basin, so nothing here is scored. What is reported is the disagreement between the instruments a regulator would reach for today, over one aquifer, from public data.

Centre pivots delineated from the annual maximum MODIS NDVI above 0.40: **2,541 km2** in 2015, against 2,494 km2 delineated at 30 m by LÃ³pez Valencia et al. 2020, HESS 24, 5251-5277. Across thresholds 0.35 to 0.50 the extent runs 2,139 to 2,749 km2.

| account | 2015 | 2019 | 2021 |
|---|---:|---:|---:|
| PML V2, 500 m | 2,110 | 1,579 | 1,422 |
| crop coefficient from NDVI x reference ET | 1,436 | 1,378 | 1,395 |
| WaPOR v2, 250 m | 622 | 409 | 445 |
| TerraClimate water balance, 4 km | 38 | 38 | 4 |
| WaPOR v3, 326 m | not published | 401 | 431 |
| **reference evapotranspiration, climatic benchmark** | **2,025** | **2,091** | **2,125** |

The three retrievals that publish a value for 2015 span a factor of **3.4**, and across the three years measured the spread runs 3.3 to 3.9. TerraClimate is left out of that range on purpose: at 38 mm/yr it is 2 per cent of the reference, because it carries no irrigation term, so it does not disagree about the agriculture, it cannot see it. At an efficiency of 0.80 the 2015 spread is 1,976 to 6,699 Mm3/yr against a published 5,500.

**The gravimetric leg needs a control and has never had one.** The Saq footprint falls at -10.93 cm/decade over 238 months. The same trend over deserts with no irrigation:

| control | cm/decade | differenced | local share | Mm3/yr |
|---|---:|---:|---:|---:|
| central Arabian shield, 42-46E 22-26N | -10.18 | -0.76 | 7% | 162 |
| An-Nafud and eastern shield, 42-46E 27-30N | -9.74 | -1.19 | 11% | 256 |
| Rub' al Khali, 47-52E 18.5-22.5N | -3.32 | -7.61 | 70% | 1,631 |
| western Rub' al Khali, 45-49E 17-21N | -3.20 | -7.74 | 71% | 1,658 |

The local share of the raw trend is between 7 and 71 per cent depending on which desert is the control, so the storage loss it implies runs 162 to 1,658 Mm3/yr over 214,266 km2. Against a crop-coefficient consumptive use of 3,648 Mm3/yr over the pivots, the two satellite legs disagree by a factor of 2.2 to 23. Neither leg can settle it alone, which is what the closure is for.

## The mascon gain: identifiability, propagation, and the Saq

The gravity operator reads `G(t) = alpha * dS(t) / A + external(t) + eps`,
and `alpha` multiplies the quantity that leg exists to supply. Two
questions the ablation grid cannot answer, because it varies which legs
are assimilated and never the gain. Reproduce with `make gain`.

**Is the gain identifiable?** Share of its prior variance the posterior
removes, by observing set.

| observations | posterior gain | prior variance removed | gravity leg |
|---|---:|---:|---|
| heads only | 0.850 | 2% | no |
| evapotranspiration + heads | 0.850 | 2% | no |
| evapotranspiration + deformation | 0.850 | 2% | no |
| evapotranspiration + gravity | 0.816 | 51% | yes |
| heads + gravity + deformation | 0.832 | 18% | yes |
| gravity only | 0.786 | 69% | yes |
| all four, coupled closure | 0.809 | 28% | yes |

Every set without a gravity leg removes at most 2 per cent, and the four-way closure removes 28. The gain is constrained only by the leg it multiplies, so the absolute scale is a prior and this is the number that says so.

**What does that prior cost?** The four-leg row at a sequence of
gain prior widths, nothing else changed.

| gain prior | residual gain error | MAE, Mm3/yr | basin abstraction bias | district 90% interval, Mm3/yr |
|---|---:|---:|---:|---:|
| the gain treated as known, plus or minus 0.001 | +0.070 | 13.94 | -8.0% | 62.4 |
| half the published uncertainty, plus or minus 0.020 | +0.055 | 9.91 | -5.5% | 65.5 |
| the published configuration, plus or minus 0.040 | +0.029 | 6.72 | -1.3% | 70.3 |
| twice the published uncertainty, plus or minus 0.080 | +0.006 | 8.09 | +1.8% | 72.9 |
| the gain and the external trend both free, free across the box | +0.005 | 9.89 | +2.5% | 73.4 |

The twin's true gain is 0.78 against a shipped prior of 0.85 plus or minus 0.04, wrong by 1.75 of its own standard deviations by construction, so the sweep measures what a wrongly computed gain costs. Believing it exactly puts **6.7 percentage points** into the basin scale that the published width buys back, at the price of a district interval 13 per cent wider. Releasing the gain and the external mass trend together moves the scale by **3.8** points and raises the error by 47 per cent, while the gain itself comes back within 0.005 of truth and the external trend takes up the discrepancy instead. A well-recovered gain is not evidence that the gravity leg has been read correctly.

**The gain on the target basin, computed rather than assumed.** The mascon polygons are recovered from the published product, which is piecewise constant on them: 44 mascons over the window, median area 112,419 km2 against the 111,279 km2 of the three-degree equal-area design. The source is the 2,202 km2 above NDVI 0.40 in every one of 2015, 2019, 2021. Reproduce with `make saq-gain`.

| reporting footprint | area, km2 | gain, source spread 0 km | gain, source spread 25 km | gain, source spread 50 km | gain, source spread 100 km | gain, source spread 150 km |
|---|---:|---:|---:|---:|---:|---:|
| the Al Jawf pivot box | 28,867 | 0.136 | 0.159 | 0.174 | 0.244 | 0.362 |
| the tight Saq box | 58,830 | 0.218 | 0.264 | 0.287 | 0.353 | 0.456 |
| the Saq box | 215,184 | 0.636 | 0.659 | 0.675 | 0.696 | 0.731 |
| the mascons that carry the pivots | 340,501 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Over whole mascons the averaging returns every unit of mass that is there, for any source geometry inside it: that footprint is 3 mascons and 340,501 km2. Under a persistent delineation the threshold moves the gain over the Saq box by 0.003; delineating from a single year moves it by 0.092. The uncertainty the product publishes over that box is 20.4 mm against the 14 mm the L0 twin generates at.

Repeating the four-leg row with the gravity leg degraded to 20.4 mm and the estimator told about it: MAE 6.72 to 6.74 Mm3/yr, basin scale -1.3 to -1.0 per cent, 90 per cent coverage 0.99 to 0.99.

## The external mass trend: the other nuisance in the gravity operator

The entry constrains the linear part of the external mass term to plus or minus 1.0 mm/yr. The L3 control boxes, unirrigated desert over the Arabian shield and the Rub' al Khali, carry 3.2, 3.3, 9.7, 10.2 mm/yr in magnitude, against 10.9 over the Saq itself, so the target basin does not support a prior that tight. Reproduce with `make drift`.

| external trend prior, mm/yr | posterior trend | MAE, Mm3/yr | basin abstraction bias | district 90% interval, Mm3/yr | 90% coverage |
|---|---:|---:|---:|---:|---:|
| the shipped constraint, plus or minus 1 | +0.45 | 6.72 | -1.3% | 70.3 | 0.99 |
| widened to the control scale, plus or minus 4 | +1.16 | 6.83 | -0.5% | 71.1 | 0.99 |
| widened past the largest control, plus or minus 10 | +1.21 | 6.88 | -0.2% | 71.3 | 0.99 |

Widening the prior 10-fold costs 2 per cent of the error, 6.72 to 6.88 Mm3/yr, moves the basin scale -1.3 to -0.2 per cent and leaves the coverage at 0.99. The constraint the target basin's controls do not support is not the one the answer rests on.

**The two nuisances as a two-by-two.** Each held at the prior the entry ships, or released.

| mascon gain | external mass trend | MAE, Mm3/yr | basin abstraction bias |
|---|---|---:|---:|
| held at the shipped prior | held at the shipped prior | 6.72 | -1.3% |
| held at the shipped prior | released, sd 10 mm/yr | 6.88 | -0.2% |
| released, sd 0.50 | held at the shipped prior | 9.13 | +1.6% |
| released, sd 0.50 | released, sd 3 mm/yr | 9.89 | +2.5% |

The pair is not symmetric. Released on its own the external trend raises the district error by 2.4 per cent; released on its own, with the trend still held, the gain raises it by 36; released together, 47. The gain carries the absolute scale and the trend does not, so the gain is the constraint that has to be defended, and it is the one that is computable on the target basin. In both released-gain rows the posterior gain lands within 0.005 of the withheld truth while the estimate is a third worse.

---

Reproduce with `make all` from a fresh clone. `make test` runs the guards, `make robustness` the seed and uniform-efficiency repeats, `make kansas-data && make kansas && make kansas-score` the Kansas rung, `make aljawf` the Al Jawf rung, and `make gain`, `make saq-gain` and `make drift` the three sensitivity studies on the gravity leg.
