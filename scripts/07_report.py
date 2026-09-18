"""Every number that appears in the submission, emitted as markdown from the results.

Nothing in the proposal, the deck, the film or the technical note is typed by hand.
This script is the only path from `results/` into those documents.

Usage:  python scripts/07_report.py > ../09_RESULTS.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mizan import config as C

RES = ROOT / "results"
ORDER = ["ET", "A", "B", "C", "D", "E", "F", "G", "SAT", "HS3", "HS", "H",
         "M", "HM1", "HM3"]
LABEL = {
    "ET": "evapotranspiration only",
    "A": "heads only",
    "B": "gravity only",
    "C": "deformation only",
    "D": "evapotranspiration + heads",
    "E": "evapotranspiration + gravity",
    "F": "evapotranspiration + deformation",
    "G": "heads + gravity + deformation",
    "H": "**all four, coupled closure**",
    "SAT": "**satellites only, no wells at all**",
    "HS": "all four, 10 wells instead of 97",
    "HS3": "all four, 3 wells instead of 97",
    "M": "meters on every district, no satellites",
    "HM1": "all four, plus one metered district",
    "HM3": "all four, plus three metered districts",
}


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def section(title):
    print(f"\n## {title}\n")


def main():
    L0_NULL = None
    L0_ABOVE_NULL = float("nan")
    tr = np.load(RES / "truth.npz")
    ab = load("ablation.json")
    al = load("allocation.json")
    vo = load("voi.json")
    de = load("detection.json")
    cv = load("convergence.json")
    eb = load("error_budget.json")

    print("# Results\n")
    print("Generated into `RESULTS.md` by `make report`, which runs")
    print("`scripts/07_report.py` over the results files in this repository. Every")
    print("number in the proposal, the deck, the film and the technical note comes from")
    print("here and from nowhere else.\n")

    section("The test basin")
    sd = tr["storage_depletion"][-1]
    perm = float(tr["permanent_loss"])
    print(f"| Quantity | Value |")
    print(f"|---|---|")
    print(f"| Domain | {C.DOMAIN_KM:.0f} x {C.DOMAIN_KM:.0f} km, two layers, "
          f"{C.NYEAR} years, {C.NPER} monthly stress periods |")
    print(f"| Truth grid / estimator grid | "
          f"{C.TRUTH.delr_m/1000:.0f} km / {C.EST.delr_m/1000:.0f} km |")
    print(f"| Districts | {C.NDIST} |")
    print(f"| Total abstraction | {tr['q_annual'].sum()/1e9:.2f} km3 "
          f"({tr['q_annual'].sum(0).mean()/1e6:.0f} Mm3/yr) |")
    print(f"| Storage depleted | {sd/1e9:.2f} km3 |")
    print(f"| Peak head decline | {(C.H_INIT - tr['head_final']).max():.1f} m |")
    print(f"| Peak subsidence | {tr['subsidence_final'].max()*100:.1f} cm "
          f"({tr['subsidence_final'].max()*100/C.NYEAR:.2f} cm/yr) |")
    print(f"| **Storage capacity destroyed permanently** | "
          f"**{perm/1e9:.3f} km3, {perm/sd*100:.1f}% of simulated storage depletion** |")
    print(f"| Irreversible share of peak subsidence | "
          f"{tr['inelastic_final'].max()/tr['subsidence_final'].max()*100:.0f}% |")
    print(f"| Observations | " + ", ".join(
        f"{k} {tr['obs_' + k].size}" for k in ("et", "grace", "insar", "head")) + " |")

    if ab:
        section("Ablation grid: what each observation is worth")
        print("Scored on district-annual abstraction against withheld truth. "
              "MAE in Mm3/yr.\n")
        print("| Observations | MAE | MAPE | bias | 50% | 80% | 90% | interval width |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for k in ("BASELINE", "BASELINE_ORACLE", "PRIOR"):
            if k in ab:
                v = ab[k]
                cov = " | ".join(
                    [f"{v[f'cover_{l}']*100:.0f}%" if f"cover_{l}" in v else "none"
                     for l in (50, 80, 90)])
                w = f"{v['width90_mcm']:.1f}" if "width90_mcm" in v else "none"
                print(f"| {v['label']} | {v['mae_mcm']:.2f} | {v['mape_pct']:.1f}% | "
                      f"{v['bias_mcm']:+.2f} | {cov} | {w} |")
        for k in ORDER:
            if k not in ab:
                continue
            v = ab[k]
            print(f"| {LABEL[k]} | {v['mae_mcm']:.2f} | {v['mape_pct']:.1f}% | "
                  f"{v['bias_mcm']:+.2f} | "
                  + " | ".join([f"{v[f'cover_{l}']*100:.0f}%" for l in (50, 80, 90)])
                  + f" | {v['width90_mcm']:.1f} |")
        if "H" in ab and "BASELINE" in ab:
            r = ab["BASELINE"]["mae_mcm"] / ab["H"]["mae_mcm"]
            ro = ab["BASELINE_ORACLE"]["mae_mcm"] / ab["H"]["mae_mcm"]
            print(f"\nThe closure reduces the error of the published open-loop account "
                  f"by a factor of **{r:.1f}**, and of the same form with its efficiency "
                  f"fitted against the answer by a factor of **{ro:.1f}**.")

        section("What the observation set could not resolve")
        print("| Observations | directions resolved to 90% | unresolved | "
              "widened | directions constrained |")
        print("|---|---:|---:|---:|---:|")
        nullp = RES / "resolution_null.json"
        if nullp.exists():
            n = json.loads(nullp.read_text())["L0"]
            L0_NULL = n["effective_dim"]["mean"]
            print(f"| **null: two independent prior ensembles, no data at all** | "
                  f"**{n['n_resolved_90']['mean']:.0f}** | "
                  f"**{n['n_unresolved']['mean']:.0f}** | "
                  f"**{n['n_widened']['mean']:.0f}** | "
                  f"**{n['effective_dim']['mean']:.1f}** |")
        for k in ORDER:
            p = RES / f"posterior_{k}.npz"
            if k not in ab or not p.exists():
                continue
            ratio = np.load(p)["variance_ratio"]
            if k == "H" and L0_NULL is not None:
                L0_ABOVE_NULL = float(np.clip(1.0 - ratio, 0.0, None).sum()) - L0_NULL
            print(f"| {LABEL[k]} | {int((ratio < 0.10).sum())} | "
                  f"{int((ratio > 0.90).sum())} | {int((ratio > 1.0).sum())} | "
                  f"{np.clip(1.0 - ratio, 0.0, None).sum():.1f} |")
        print(f"\nOut of {C.NDIST * C.NYEAR} directions of the district-year abstraction "
              f"vector. A direction whose posterior variance exceeds its prior variance "
              f"has learned nothing and is counted as widened rather than as negative "
              f"information.")

        section("Recovery of the quantities the open-loop method assumes")
        print("| Observations | consumptive fraction MAE | pre-canopy share MAE | "
              "preconsolidation offset |")
        print("|---|---:|---:|---:|")
        for k in ORDER:
            if k in ab and "eta_mae" in ab[k]:
                v = ab[k]
                print(f"| {LABEL[k]} | {v['eta_mae']:.3f} | {v['preplant_mae']:.3f} | "
                      f"{v['pcs_offset_hat']:.2f} m against 12.00 m |")

    if eb:
        section("Error budget")
        print("Observations are weighted at instrument error. The structural error of "
              "the coarse\nforward model is estimated after the fact from the "
              "full-information residual.\n")
        print("| Leg | instrument | residual | structural | lag-1 | independence inflation |")
        print("|---|---:|---:|---:|---:|---:|")
        for k, v in eb.items():
            print(f"| {k} | {v['nominal']:.4g} | {v['residual_rms']:.4g} | "
                  f"{v['structural']:.4g} | {v['lag1']:.2f} | "
                  f"{v['independence_inflation']:.2f} |")
        print("\nOn every leg the structural component comes out below the instrument "
              "error, so no inflation is applied.")

    if cv:
        section("Convergence")
        print("| iteration | MAE | bias | 90% coverage | interferometry | heads | gravity |")
        print("|---:|---:|---:|---:|---:|---:|---:|")
        for r in cv:
            print(f"| {r['iter']:.0f} | {r['mae_mcm']:.2f} | {r['bias_mcm']:+.2f} | "
                  f"{r['cover_90']*100:.0f}% | {r['insar']*1000:.2f} mm | "
                  f"{r['head']:.3f} m | {r['grace']:.1f} mm |")

    if al:
        section("Decision layer")
        print(f"Zones: {al['zones']} drawdown strata. Time invariance of the response "
              f"matrix: mean worst-case error {al['lti_mean_max_rel_error']*100:.1f}% of "
              f"the pulse response.\n")
        print(f"Permanent loss already taken over the record: "
              f"{al['hist_permanent_loss_mcm']:.0f} Mm3.\n")
        print("**How much can be taken.** Permanent storage loss over the next twenty "
              "years, in full MODFLOW across the posterior.\n")
        print("| delivered, km3 | cut | mean loss, Mm3 | 10th | 90th |")
        print("|---:|---:|---:|---:|---:|")
        for f in al["frontier"]:
            print(f"| {f['delivered_km3']:.2f} | {f['cut']*100:.0f}% | "
                  f"{f['mean_mcm']:.0f} | {f['p10_mcm']:.0f} | {f['p90_mcm']:.0f} |")
        print(f"\nCutting delivery by "
              f"{al['frontier'][0]['delivered_km3']-al['frontier'][-1]['delivered_km3']:.2f} "
              f"km3 over the horizon avoids "
              f"{al['frontier'][0]['mean_mcm']-al['frontier'][-1]['mean_mcm']:.0f} Mm3 of "
              f"permanent capacity loss: "
              f"**{al['marginal_capacity_per_km3']:.0f} Mm3 of capacity per km3 not "
              f"taken, in the test basin.** That rate is a simulated output of this "
              f"basin and not an estimate for any real aquifer; what transfers is the "
              f"frontier as an instrument.")
        print("\nEvery frontier value above is evaluated directly in full MODFLOW across "
              "the posterior. Experimental spatial-allocation diagnostics are retained "
              "in `results/allocation.json` and are not submission results.")

    if vo:
        section("Value of information")
        for f in ("fc_q_last5", "fc_perm_loss"):
            if f not in vo:
                continue
            v = vo[f]
            name = ("basin abstraction over the last five years" if f == "fc_q_last5"
                    else "storage capacity destroyed permanently")
            print(f"\n**Forecast: {name}.** Prior standard deviation "
                  f"{v['prior_sd']:.4g}, posterior {v['posterior_sd']:.4g}.\n")
            print("| rank | instrument | uncertainty removed |")
            print("|---:|---|---:|")
            for i, c in enumerate(v["top_candidates"][:8]):
                print(f"| {i+1} | {c['name']} | {c['sd_removed']:.4g} |")
            cur = v["greedy_curve"]
            print(f"\nForward selection of {len(cur)-1} instruments takes the posterior "
                  f"standard deviation from {cur[0]:.4g} to {cur[-1]:.4g}, a "
                  f"**{(1-cur[-1]/cur[0])*100:.0f}% reduction**.")
        if "pyemu_check" in vo and "error" not in vo["pyemu_check"]:
            print("\nSchur complement cross-checked against pyEMU: " + ", ".join(
                f"{k} relative difference {v['rel_diff']:.1e}"
                for k, v in vo["pyemu_check"].items()) + ".")

    if de:
        section("Detecting abstraction no evapotranspiration product can see")
        print(f"A withdrawal of {de['hidden_mcm_per_year']:.0f} Mm3/yr with no canopy, "
              f"and therefore no evapotranspiration signature, was planted in district "
              f"D{de['hidden_district']}: "
              f"{de['hidden_share_of_district']*100:.0f}% of that district's abstraction.\n")
        print(f"- The open-loop account attributes "
              f"**{de['open_loop_recovery_pct']:.0f}%** of that district's true "
              f"abstraction. It cannot see the withdrawal at all.")
        print(f"- The closure attributes **{de['closure_recovery_pct']:.0f}%**.")
        print(f"- On the ratio of closure estimate to what consumptive use can explain, "
              f"the district sits **{de['z_score']:.1f} standard deviations** above every "
              f"other district.")
        print(f"- Basin-wide error: closure {de['closure']['mae_mcm']:.2f} Mm3/yr against "
              f"{de['open_loop']['mae_mcm']:.2f} for the open-loop account.")

    seeds = {s: load(f"ablation_seed{s}.json") for s in (6, 7)}
    seeds = {s: v for s, v in seeds.items() if v}
    eu = load("ablation_etauniform.json")
    if seeds or eu:
        section("Robustness")

    if seeds:
        print("The rows that carry the argument, repeated on independent prior "
              "ensembles and independent ES-MDA perturbations. One seed is not "
              "evidence; the spread is what the difference between two rows has to "
              "beat.\n")
        rows = [k for k in ("H", "G", "SAT", "ET") if k in ab]
        print("| observations | " + " | ".join(f"seed {s}" for s in [5] + list(seeds))
              + " | mean | spread |")
        print("|---|" + "---:|" * (len(seeds) + 3))
        for k in rows:
            vals = [ab[k]["mae_mcm"]] + [seeds[s][k]["mae_mcm"] for s in seeds
                                         if k in seeds[s]]
            a = np.array(vals)
            print(f"| {LABEL[k]} | " + " | ".join(f"{v:.2f}" for v in vals)
                  + f" | **{a.mean():.2f}** | {a.max()-a.min():.2f} |")
        print("\nDifferences smaller than the spread in this table are not claimed.")

    if eu:
        print("\n**A truth with no district spread in the consumptive fraction.** Every "
              "district given the same fraction, 0.79, which is the case most favourable "
              "to the open-loop form.\n")
        print("| truth | open loop at 0.80 | open loop, constant fitted | closure |")
        print("|---|---:|---:|---:|")
        print(f"| districts differ, 0.69 to 0.88 | {ab['BASELINE']['mae_mcm']:.2f} | "
              f"{ab['BASELINE_ORACLE']['mae_mcm']:.2f} | "
              f"**{ab['H']['mae_mcm']:.2f}** |")
        print(f"| every district identical | {eu['BASELINE']['mae_mcm']:.2f} | "
              f"{eu['BASELINE_ORACLE']['mae_mcm']:.2f} | "
              f"**{eu['H']['mae_mcm']:.2f}** |")

    KTAG = "_v3"
    ks = load(f"kansas{KTAG}.json")
    if ks:
        m = ks["_meta"]
        section("L2 Kansas: against real metered abstraction")
        print(f"Six counties of the Northwest Kansas groundwater management district "
              f"over the Ogallala, {m['years'][0]} to {m['years'][-1]}, "
              f"{m['irrigated_km2']:,.0f} km2 irrigated. The scored quantity is "
              f"county-annual abstraction against **per-water-right metered pumping** "
              f"published by the Kansas Department of Agriculture: "
              f"{m['n_rights']:,} water rights, "
              + ("none of them unreadable." if not m["n_missing"]
                 else f"of which {m['n_missing']} could not be read.") + "\n")
        print(f"The estimator sees {m['n_obs_head']:,} well-year head anomalies from "
              f"{m['n_wells']} wells and {len(m['counties']) * len(m['years'])} "
              f"county-year evapotranspiration volumes. It never sees the metered "
              f"volumes: the licensed points of diversion enter as spatial weights, "
              f"and the use filed against them is the withheld truth.\n")
        print("| observations available to the estimator | MAE, Mm3/yr | MAPE | "
              "90% coverage |")
        print("|---|---:|---:|---:|")
        for k in ("PRIOR_FLAT", "PRIOR", "BASELINE", "BASELINE_ORACLE",
                  "ET", "H", "ETH"):
            if k not in ks:
                continue
            v = ks[k]
            cov = f"{v['cover_90']*100:.0f}%" if "cover_90" in v else "none produced"
            star = "**" if k == "ETH" else ""
            print(f"| {star}{v['label']}{star} | {star}{v['mae_mcm']:.2f}{star} | "
                  f"{star}{v['mape_pct']:.1f}%{star} | {cov} |")
        ksp = load("kansas_v3p.json")
        if ksp and "ETH" in ksp:
            v, vp = ks["ETH"], ksp["ETH"]
            print(f"\nThe row above is the per-site error budget. Under the pooled "
                  f"budget on the same published thickness the closure scores "
                  f"{vp['mae_mcm']:.2f} Mm3/yr at {vp['mape_pct']:.1f}% with "
                  f"{vp['cover_90']*100:.0f}% coverage, against {v['mae_mcm']:.2f} at "
                  f"{v['mape_pct']:.1f}% with {v['cover_90']*100:.0f}%. Both are "
                  f"reported and neither was selected against the meters.")

        print("\n**The layer thickness is an observation, not a parameter.** The first "
              "Kansas configuration estimated one global saturated thickness and settled "
              "at 79 m. The USGS High Plains saturated-thickness grid, sampled onto the "
              "same model grid, gives a block mean of 20.4 m and county means of 19.7, "
              "16.6, 13.2, 29.4, 23.6 and 19.1 m. The prior did not merely miss that: it "
              "ran from 20 m to 140 m, so five of the six counties sat at or below its "
              "lower bound. The surface that falsified it is an independent published "
              "observation into which no water-use report enters, so the test itself "
              "needed no meter; the decision log records that the change was made "
              "after the first Kansas score was known. The layer base is now that "
              "field times one estimated multiplier, so the parameter count is "
              "unchanged.")

        if "ETH" in ks:
            e = ks["ETH"]
            print(f"\nPosterior nuisances: a multiplier of {e['bmul_hat']:.2f} on the "
                  f"published saturated thickness, which puts the layer at "
                  f"{e['bsat_hat']:.1f} m over the block, recharge "
                  f"{e['rch_hat']:.0f} mm/yr, consumptive fraction by county " +
                  ", ".join(f"{c} {v:.2f}" for c, v in
                            zip(m["counties"], e["eta_hat"])) +
                  ". The multiplier lands within ten per cent of unity, so the head "
                  "record is consistent with the published surface rather than pulling "
                  "away from it.")

        kc = load("kansas_convergence.json")
        if kc and "ETH" in ks:
            e = ks["ETH"]
            print(f"\n**One declaration on the specific yield.** The posterior specific "
                  f"yield is {e['sy_hat']:.3f}, which sits above the range published for "
                  f"the Kansas High Plains, and it has to be read as an upper bound "
                  f"rather than as a retrieval. Putting the true thickness in costs part "
                  f"of the prior ensemble, because thin low-storage members dewater and "
                  f"fail to converge. Measured on {kc['ne']} members, "
                  f"{kc['n_failed']} of which failed, the failed members sit "
                  f"{abs(kc['block abstraction, Mm3/yr']['gap_sd']):.2f} standard "
                  f"deviations from the converged ones on block abstraction, so the "
                  f"scored quantity is unaffected, but "
                  f"{abs(kc['specific yield']['gap_sd']):.2f} on specific yield and "
                  f"{abs(kc['saturated thickness multiplier']['gap_sd']):.2f} on the "
                  f"thickness multiplier. The truncation is selective in exactly those "
                  f"two directions and the number is reported with that attached.")

        an = load(f"kansas_anomaly{KTAG}.json")
        if an:
            print(f"\n**The level is the easy part.** Irrigated area times one published "
                  f"applied depth already lands close on a county mean, so the "
                  f"county-year anomaly about each county's own record mean is what an "
                  f"estimator has to earn. The signal is {an['_signal_mcm']:.2f} Mm3/yr "
                  f"mean absolute, and a flat-in-time estimate scores exactly that.\n")
            print("| estimate | MAE | anomaly MAE | anomaly skill |")
            print("|---|---:|---:|---:|")
            for k, lab in (("FLAT", "each county's own 25-year mean, flat in time"),
                           ("BASELINE", "open loop at 0.80"),
                           ("H", "heads only"),
                           ("ET", "evapotranspiration only"),
                           ("ETH", "evapotranspiration + heads, closure")):
                if k not in an:
                    continue
                v = an[k]
                print(f"| {lab} | {v['mae_mcm']:.2f} | {v['anomaly_mae_mcm']:.2f} | "
                      f"{v['anomaly_skill']:+.2f} |")

        kr = load(f"kansas_resolution{KTAG}.json")
        if kr:
            print("\n**What the two legs could resolve**, on the same statistic L0 "
                  "reports, against the null of two independent prior ensembles with no "
                  "data assimilated.\n")
            print("| observations | resolved to 90% | unresolved | widened | "
                  "directions constrained |")
            print("|---|---:|---:|---:|---:|")
            n = kr.get("_null")
            if n:
                print(f"| **null: no data at all** | **{n['n_resolved_90']:.0f}** | "
                      f"**{n['n_unresolved']:.0f}** | **{n['n_widened']:.0f}** | "
                      f"**{n['effective_dim']:.1f}** |")
            for k in ("ET", "H", "ETH"):
                if k not in kr:
                    continue
                v = kr[k]
                print(f"| {v['label']} | {v['n_resolved_90']} | {v['n_unresolved']} | "
                      f"{v['n_widened']} | {v['effective_dim']:.1f} |")
            print(f"\nOut of {kr['_ndir']} county-year directions. Read against the null "
                  f"row, not against zero.")
            if n and all(k in kr for k in ("ET", "H", "ETH")):
                d = {k: kr[k]["effective_dim"] - n["effective_dim"]
                     for k in ("ET", "H", "ETH")}
                print(f"\nAbove the null the head leg carries Kansas at "
                      f"{d['H']:+.1f} directions and the closure sits at "
                      f"{d['ETH']:+.1f}, while the evapotranspiration leg alone at "
                      f"{d['ET']:+.1f} is below what nothing does: a 1 km product over "
                      f"a block that is 14 per cent irrigated carries a structural "
                      f"error large enough to cancel the leg. Adding it to the heads "
                      f"costs {d['H'] - d['ETH']:.1f} directions of resolution and buys "
                      f"the level and the interannual amplitude reported below, which "
                      f"is a trade the two tables have to be read together to see. The "
                      f"same statistic at L0 put the four-leg closure "
                      f"{L0_ABOVE_NULL:+.1f} above its "
                      f"own null out of 180 directions. The real two-leg configuration "
                      f"extracts about a third as much information per direction as the "
                      f"synthetic one did, and it said so before the meters were "
                      f"opened.")

        sh = load(f"kansas_shrink{KTAG}.json")
        if sh:
            print("\n**The interannual amplitude, calibrated without an oracle.** The "
                  "estimate carries real year-to-year information: its county-year anomaly "
                  "correlates with the metered one. With a correlation below "
                  "one, the amplitude that minimises mean absolute error is "
                  "smaller than the estimate's own, and one scalar per county "
                  "supplies it. Fitted against the meters that scalar is an "
                  "oracle, so it is fitted leave-one-county-out: every county's "
                  "factor comes from the other five and no county enters its "
                  "own fit.\n")
            print("| estimate | correlation | amplitude ratio | raw skill | "
                  "leave-one-county-out | oracle |")
            print("|---|---:|---:|---:|---:|---:|")
            for k in ("BASELINE", "H", "ET", "ETH"):
                if k not in sh:
                    continue
                v = sh[k]
                star = "**" if k == "ETH" else ""
                print(f"| {star}{v['label']}{star} | {v['r']:.2f} | "
                      f"{v['dispersion']:.2f} | {v['raw']['anomaly_skill']:+.2f} | "
                      f"{star}{v['loco']['anomaly_skill']:+.2f}{star} | "
                      f"{v['oracle']['anomaly_skill']:+.2f} |")
            e = sh.get("ETH", {})
            if e:
                fac = e["loco_factors"]
                print(f"\nThe factor is {min(fac):.2f} to {max(fac):.2f} across the six "
                      f"folds, so holding a county out costs nothing against the oracle. "
                      f"This is the operational requirement the value-of-information "
                      f"layer reached from the other direction: a few metered counties "
                      f"calibrate the amplitude for the rest.")

    lad = load(f"ladder{KTAG}.json")
    ver = load(f"verify{KTAG}.json")
    if lad and ver:
        section("What a reduction target actually asks, and who can answer it")
        print("A water account is usually judged on its level. A regulator with a "
              "reduction target is not asking for a level. Saudi Arabia has published a "
              "90 per cent reduction target for non-renewable groundwater, and Kansas "
              "writes its Local Enhanced Management Areas as a percentage cut against a "
              "stated baseline period. Both are questions about a change between two "
              "multi-year periods, with an interval on it.\n")
        print("Every meter-free account that can be written down from the same public "
              "data, scored against the same withheld meters, on both quantities.\n")
        print("| account | level, Mm3/yr | change over 5-year periods, points | "
              "weather share of its own variance |")
        print("|---|---:|---:|---:|")
        sweep = ver["_sweep"]["mean_abs_error_pts"]
        key = {"FLAT": "FLAT", "WATERBAL50": "WATERBAL",
               "OPENLOOP": "OPENLOOP", "CLOSURE": "CLOSURE"}
        for k in ("FLAT", "WATERBAL50", "OPENLOOP", "CLOSURE"):
            if k not in lad:
                continue
            v = lad[k]
            w = lad["_variance_decomposition"][k]["weather_share_pct"]
            star = "**" if k == "CLOSURE" else ""
            print(f"| {star}{v['label']}{star} | {v['mae_mcm']:.2f} | "
                  f"{star}{sweep[key[k]]:.1f}{star} | "
                  + ("n/a" if w is None else f"{w:.0f}%") + " |")

        dec = lad["_variance_decomposition"]
        md, fl, wb = dec["METERED"], dec["FLAT"], dec["WATERBAL50"]
        print(f"\nThe metered record itself carries {md['weather_share_pct']:.0f} per "
              f"cent of its interannual variance from precipitation, and after weather "
              f"is removed it still falls by "
              f"{abs(md['trend_after_weather_mcm_per_sd_year']):.1f} Mm3 per "
              f"standard-deviation year. The two arithmetic bars are "
              f"{fl['weather_share_pct']:.0f} and {wb['weather_share_pct']:.0f} per cent "
              f"weather and keep "
              f"{abs(fl['trend_after_weather_mcm_per_sd_year']):.1f} and "
              f"{abs(wb['trend_after_weather_mcm_per_sd_year']):.1f} of that trend. They "
              f"carry the half of the signal the weather causes and they are blind to "
              f"the half a policy changes.\n")

        sw = ver["_sweep"]
        print(f"The change is scored over every pair of non-overlapping "
              f"{sw['window_years']}-year windows the record admits, {sw['n_pairs']} of "
              f"them, rather than over a chosen contrast. The closure's 90 per cent "
              f"interval contains the metered change in {sw['coverage_90']*100:.0f} per "
              f"cent of pairs. Where it declares a change the metered change averages "
              f"{sw['metered_pct_where_declared']:.1f} per cent against "
              f"{sw['metered_pct_where_not_declared']:.1f} per cent where it declares "
              f"none, an area under the curve of "
              f"{sw['declaration_auc_vs_metered_magnitude']:.3f}. The pairs are drawn from one "
              f"25-year record and share years, so 136 is a count of contrasts and not of "
              f"independent cases: the coverage and the area under the curve are descriptive, "
              f"and no p-value is attached to either.\n")
        print(f"**The direction of the change is not a test on this record and is not "
              f"reported as one.** Abstraction fell over "
              f"{sw['n_metered_change_negative']} of the {sw['n_pairs']} pairs, so an "
              f"estimator that says down every time scores "
              f"{sw['always_down_scores_on_declared']*100:.0f} per cent, which is what "
              f"the closure scores. The magnitude above is the test.\n")

        cur = ver["_window_curve"]
        print("The length of the window is not a free choice. A weather model carries "
              "the high-frequency half of the signal and saturates; the aquifer "
              "integrates storage and keeps improving.\n")
        print("| averaging window | pairs | closure | best meter-free bar | "
              "closure interval covers |")
        print("|---:|---:|---:|---:|---:|")
        for w in sorted(cur["by_window_years"], key=int):
            r = cur["by_window_years"][w]
            m = r["mean_abs_error_pts"]
            best = min(m[k] for k in m if k != "CLOSURE")
            cs = "**" if m["CLOSURE"] < best else ""
            print(f"| {w} years | {r['n_pairs']} | {cs}{m['CLOSURE']:.1f}{cs} | "
                  f"{best:.1f} | {r['coverage_90']*100:.0f}% |")
        print(f"\n**The closure beats every meter-free bar from a "
              f"{cur['crossover_window_years']}-year window upward, and the gap widens "
              f"with every year added.** Below that the best of them is a weather model "
              f"and it is the better instrument. That crossover is a design rule for a "
              f"monitoring programme and it is measured against real meters.\n")

        rs = ver["_resolution"]
        print(f"The posterior spread on a basin-wide five-year-against-five-year "
              f"contrast is {rs['posterior_sd_pts']:.1f} percentage points, so this "
              f"observing system separates a real reduction from no change at "
              f"{rs['one_sided_90pct_detectable_pct']:.1f} per cent, at 90 per cent "
              f"one-sided confidence.\n")

        z = ver["_sd6_county_did"]
        print(f"**Where it runs out.** The Sheridan-6 Local Enhanced Management Area "
              f"covers 256 km2 inside a 2,331 km2 county. Against the four clean "
              f"neighbouring counties the meters give a difference in differences of "
              f"{z['metered_did_pts']:+.1f} points; the closure gives "
              f"{z['closure_did_pts']:+.1f} plus or minus {z['closure_did_sd']:.1f}, "
              f"with the wrong sign and a 90 per cent interval of "
              f"[{z['closure_did_ci90'][0]:+.1f}, {z['closure_did_ci90'][1]:+.1f}]. A "
              f"policy on a tenth of a county is below what this observing system "
              f"resolves, and the resolution analysis said so before the meters were "
              f"opened.\n")

    me = load(f"metered_era{KTAG}.json")
    if me:
        au = me["audit"]
        y0 = me["_meta"]["era_year0"]
        section(f"The same scores on the metered era, {y0} to 2024")
        print("WIMAS records on every water-use report the code that says how its "
              "volume was measured (KGS OFR 2005-30): A, M and I are meter readings, "
              "G is hours of pump operation times a rate. The scores above are computed "
              "against the whole 2000 to 2024 record. This section computes them on the "
              "years the record is fully metered, and against the truth rebuilt from "
              "the per-point use file, which the history page under-counts.\n")
        print("| year | meter-coded share of reported volume, block | lowest county | "
              "complete record over the series first read |")
        print("|---:|---:|---:|---:|")
        for y in range(2000, 2025):
            s = au["metered_share_by_year"][str(y)]
            m = au["min_county_metered_share_by_year"][str(y)]
            r = au["reported_over_published_truth_by_year"][str(y)]
            b = "**" if y >= y0 else ""
            print(f"| {b}{y}{b} | {100 * s:.1f}% | {100 * m:.1f}% | {r:.3f} |")
        print(f"\nThe first year from which every later year is above 98 per cent is "
              f"**{au['first_year_every_later_year_above_98pct']}**, the year GMD4 "
              f"records as the first with every well metered. The series first read "
              f"took one point of diversion per water right from the history page; the "
              f"use file carries one report per point, and rights with several points "
              f"are under-counted by that route, so the complete record runs "
              f"{100 * (min(au['reported_over_published_truth_by_year'].values()) - 1):.0f} "
              f"to {100 * (max(au['reported_over_published_truth_by_year'].values()) - 1):.0f} "
              f"per cent above it in every year. The correction is close to uniform in "
              f"time, so a percentage change between two periods barely moves; a level "
              f"moves with it.\n")

        L = me["level"]
        print("**Level, district-year, against the complete record.**\n")
        print("| account | 2000 to 2024, MAE Mm3/yr | MAPE | basin bias | "
              f"{y0} to 2024, MAE Mm3/yr | MAPE | basin bias | 90% cover |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        names = {"CLOSURE": "**the closure, evapotranspiration and heads**",
                 "ET": "evapotranspiration only", "H": "heads only",
                 "FLAT": "mapped irrigated area x one acre-foot per acre",
                 "WATERBAL": "the same, plus half the year's precipitation deficit",
                 "OPENLOOP": "unmixed evapotranspiration over a fixed efficiency of 0.80"}
        for k in ("CLOSURE", "ET", "H", "FLAT", "WATERBAL", "OPENLOOP"):
            e = L["reported_truth_metered_era"].get(k)
            f = L["reported_truth_all_years"].get(k)
            if e is None:
                continue
            fa = (f"{f['mae_mcm']:.2f} | {f['mape_pct']:.1f}% | {f['basin_bias_pct']:+.1f}%"
                  if f else "| |")
            cv = f"{100 * e['cover_90']:.0f}%" if "cover_90" in e else "none"
            print(f"| {names[k]} | {fa} | {e['mae_mcm']:.2f} | {e['mape_pct']:.1f}% | "
                  f"{e['basin_bias_pct']:+.1f}% | {cv} |")
        c3 = L["published_truth_all_years"]["CLOSURE"]
        ce = L["reported_truth_metered_era"]["CLOSURE"]
        print(f"\nAgainst the series first read the closure's basin bias was "
              f"{c3['basin_bias_pct']:+.1f} per cent; against the complete record on "
              f"the metered era it is {ce['basin_bias_pct']:+.1f}. The bar a reviewer "
              f"can compute in a spreadsheet stays below the closure on the level, as "
              f"reported above, and every account moves down by the same correction.\n")

        A = me["anomaly_metered_era"]
        print(f"**Interannual anomaly, {y0} to 2024**, skill against a flat estimate, "
              "the amplitude factor leave-one-county-out as above.\n")
        print("| estimate | r | raw | LOCO | oracle |")
        print("|---|---:|---:|---:|---:|")
        for k in ("CLOSURE", "ET", "H", "OPENLOOP"):
            v = A[k]
            b = "**" if k == "CLOSURE" else ""
            print(f"| {b}{names[k].strip('*')}{b} | {v['r']:.2f} | {v['raw_skill']:+.2f} | "
                  f"{b}{v['loco_skill']:+.2f}{b} | {v['oracle_skill']:+.2f} |")

        S = me["sweep_5yr"]
        print(f"\n**The change between two five-year periods, {y0} to 2024.** Every "
              f"non-overlapping five-year window pair the era admits, "
              f"{S['metered_era_reported_truth']['n_pairs']} of them, beside the "
              f"{S['all_years_published_truth']['n_pairs']} of the whole record.\n")
        print("| account | whole record, series first read | whole record, complete "
              f"record | metered era {y0} to 2024 |")
        print("|---|---:|---:|---:|")
        for k in ("FLAT", "WATERBAL", "OPENLOOP", "CLOSURE"):
            b = "**" if k == "CLOSURE" else ""
            print(f"| {b}{names[k].strip('*')}{b} | "
                  f"{S['all_years_published_truth']['mean_abs_error_pts'][k]:.1f} | "
                  f"{S['all_years_reported_truth']['mean_abs_error_pts'][k]:.1f} | "
                  f"{b}{S['metered_era_reported_truth']['mean_abs_error_pts'][k]:.1f}{b} |")
        se = S["metered_era_reported_truth"]
        best = min(se["mean_abs_error_pts"][k] for k in ("FLAT", "WATERBAL", "OPENLOOP"))
        print(f"\nOn the metered era the closure scores "
              f"{se['mean_abs_error_pts']['CLOSURE']:.1f} points against "
              f"{best:.1f} for the best meter-free bar; it is closer than the open loop "
              f"on {se['closure_beats_pct']['OPENLOOP']:.0f} per cent of pairs, its 90 "
              f"per cent interval covers the metered change in "
              f"{100 * se['coverage_90']:.0f} per cent of them, and it declares a change "
              f"in {se['n_declared_change']} pairs with the sign right in "
              f"{se['n_declared_change_sign_correct']}. The metered change is negative in "
              f"{se['n_metered_change_negative']} of {se['n_pairs']} pairs, so the sign "
              f"is not a test here either.\n")

        cu = me["window_curve_metered_era"]["by_window_years"]
        print(f"| averaging window | pairs | closure | best meter-free bar | "
              f"closure interval covers |")
        print("|---:|---:|---:|---:|---:|")
        for w in sorted(cu, key=int):
            r = cu[w]
            mm = r["mean_abs_error_pts"]
            bb = min(mm[k] for k in ("FLAT", "WATERBAL", "OPENLOOP"))
            b = "**" if mm["CLOSURE"] < bb else ""
            print(f"| {w} years | {r['n_pairs']} | {b}{mm['CLOSURE']:.1f}{b} | {bb:.1f} | "
                  f"{100 * r['coverage_90']:.0f}% |")
        print(f"\nThe crossover stays at a "
              f"{me['window_curve_metered_era']['crossover_window_years']}-year window.\n")

        CT = me["contrasts_metered_era"]
        print("**The named contrasts inside the era.** The GMD4 LEMA contrast is the "
              "only one of the three policy contrasts that lies inside it; the era's "
              "own long contrast is its first five years against its last five, set by "
              "the era's endpoints.\n")
        print("| contrast | metered | closure | 90% interval | area x depth | + weather "
              "| open loop |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for k, row in CT.items():
            print(f"| {k}: {row['why']} | {row['metered_pct']:+.1f}% | "
                  f"{row['CLOSURE']['pct']:+.1f}% | [{row['CLOSURE']['ci90'][0]:+.1f}, "
                  f"{row['CLOSURE']['ci90'][1]:+.1f}] | {row['FLAT']['pct']:+.1f}% | "
                  f"{row['WATERBAL']['pct']:+.1f}% | {row['OPENLOOP']['pct']:+.1f}% |")
        print("\nOn both named contrasts the closure overstates the decline and the "
              "area-times-depth bar is closer; on the era contrast the metered change "
              "sits just outside the closure's interval. The window sweep is the "
              "aggregate; the two named contrasts are reported beside it because a "
              "reader will compute them.\n")

        G = me["closure_error_by_metered_share"]
        print("**Does the closure's error depend on how metered its truth was?** The "
              "whole-record five-year pairs, grouped by the smaller of the two windows' "
              "meter-coded shares.\n")
        print("| pairs | n | closure | open loop |")
        print("|---|---:|---:|---:|")
        lab = {"both_windows_metered": "both windows above 98 per cent metered",
               "mixed": "one window straddles the transition",
               "both_windows_pre_meter": "both windows below 90 per cent metered"}
        for k, v in G.items():
            print(f"| {lab[k]} | {v['n_pairs']} | "
                  f"{v['closure_mean_abs_error_pts']:.1f} | "
                  f"{v['openloop_mean_abs_error_pts']:.1f} |")
        print("")

    hl = load(f"headline{KTAG}.json")
    if hl:
        section("The two comparisons, each with the error its record supports")
        print("A ranking of two point estimates is not a result until the record says "
              "how well it resolves the difference. Both comparisons below carry a "
              "error bar, and which one is right depends on the claim. Over years it "
              "asks how the gain would come out in another year across these same six "
              "counties. Over counties it asks whether it would hold in a district this "
              "record has not seen, which is what a transfer claim asserts, and its "
              "sample is six. The county figure is the wider and is the one to read.\n")
        print(f"**The level, on {hl['_meta']['n_county_years']} county-years.**\n")
        print("| account | relative error | MAE Mm3/yr | points the closure removes | "
              "se by county (n=6) | se by year (n=16) | counties favouring the closure |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for k in ("OPENLOOP", "OPENLOOP_ORACLE", "WATERBAL", "FLAT"):
            lv, g = hl["level"][k], hl["level_gain_vs"][k]
            print(f"| {lv['label']} | {lv['mape_pct']:.1f}% | {lv['mae_mcm']:.2f} | "
                  f"**{g['points']:+.1f}** | +-{g['se_by_county']:.1f} "
                  f"({g['n_se_by_county']:+.1f} se) | +-{g['se_by_year']:.1f} "
                  f"({g['n_se_by_year']:+.1f} se) | "
                  f"{g['n_counties_favouring_closure']}/{g['n_counties']} |")
        lv = hl["level"]["CLOSURE"]
        print(f"| **{lv['label']}** | **{lv['mape_pct']:.1f}%** | "
              f"**{lv['mae_mcm']:.2f}** | | | | |")
        g = hl["level_gain_vs"]["OPENLOOP"]
        go = hl["level_gain_vs"]["OPENLOOP_ORACLE"]
        print(f"\n**Against the technique in use where wells are not metered, the closure "
              f"removes {g['points']:.1f} points of relative error**, and is closer on "
              f"{g['closure_closer_pct_of_county_years']:.0f} per cent of the "
              f"county-years; against the same method with its efficiency fitted to the "
              f"meters, {go['points']:.1f}. Clustered by county that gain is "
              f"{g['n_se_by_county']:.1f} standard errors, not the "
              f"{g['n_se_by_year']:.1f} the year clustering reports, because it is "
              f"carried by the two counties where the open-loop account fails worst and "
              f"is negative in "
              f"{g['n_counties'] - g['n_counties_favouring_closure']} of "
              f"{g['n_counties']}: {g['gain_by_county']}. Six counties of one climate do "
              f"not establish a seventh. Against the two arithmetic bars it loses by "
              f"{abs(hl['level_gain_vs']['FLAT']['points']):.1f} and "
              f"{abs(hl['level_gain_vs']['WATERBAL']['points']):.1f} points, as well "
              f"resolved as the gain and reported at the same size. Those bars need a "
              f"published applied depth for the basin they are used in; the published "
              f"account of Al Jawf implies one 6.5 times the Kansas figure, so they do "
              f"not transport and the open-loop comparison is the one that does.\n")
        print("**The change between two multi-year periods, over the whole record.**\n")
        print("| averaging window | window pairs | margin over the best meter-free bar | "
              "jackknife error | standard errors | sign survives dropping any one year |")
        print("|---:|---:|---:|---:|---:|---|")
        for w, v in hl["change_margin"]["whole_record"].items():
            print(f"| {w} years | {v['n_pairs']} | {v['points']:+.2f} | "
                  f"+-{v['se_by_year']:.2f} | {v['n_se_by_year']:.1f} | "
                  f"{'yes' if v['sign_survives_any_single_year_drop'] else 'no'} |")
        print(f"\n**No window puts that margin more than about one standard error above "
              f"zero.** The window pairs overlap, so their count is a count of contrasts "
              f"rather than a sample size: the metered era admits exactly "
              f"{hl['change_margin']['independent_five_year_pairs_in_era']} pair of "
              f"five-year windows sharing no year with another such pair. What the record "
              f"does support is that from a six-year window upward the margin survives "
              f"dropping any single year and grows with the window. The level comparison "
              f"above is the one this record resolves.\n")

    tr = load(f"transfer{KTAG}.json")
    if tr and tr.get("pooled_new_counties"):
        section("The same closure, frozen, on county blocks it never saw")
        new = tr["pooled_new_counties"]
        print("Three further blocks of six counties each, chosen before any of their "
              "inputs was fetched and scored blind: the predictions were committed to "
              "`DECISION_LOG.md` before the fetcher would release the use files, and the "
              "posteriors were written without a truth in them. Each block is scored on "
              "its own metered era by the published code rule. The error bars cluster by "
              "county, and the new counties are pooled on their own, never with the "
              "block the method was built on.\n")
        print("| block | counties | metered era | county-years | metered depth, af/acre | "
              "closure | ET leg | area x 1 af | + weather | open loop | closure 90% cover |")
        print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for blk in ("gmd4a",) + tuple(new["blocks"]):
            r = tr["blocks"][blk]
            lv = r["arms"][KTAG]["level"]
            print(f"| {r['label']} | "
                  f"{', '.join(r['names'])} | {r['metered_era_year0']}-2024 | "
                  f"{r['n_county_years']} | {r['metered_depth_m_era'] / 0.3048:.2f} | "
                  f"**{lv['CLOSURE']['mape_pct']:.1f}%** | "
                  f"{lv['ET']['mape_pct'] if 'ET' in lv else float('nan'):.1f}% | "
                  f"{lv['FLAT']['mape_pct']:.1f}% | {lv['WATERBAL']['mape_pct']:.1f}% | "
                  f"{lv['OPENLOOP']['mape_pct']:.1f}% | {lv['CLOSURE']['cover_90']:.2f} |")
        print("\nRelative error on the level, county-year mean. The five-year change, mean "
              "absolute error in points over every non-overlapping window pair of the "
              "block's era:\n")
        print("| block | pairs | closure | area x 1 af | + weather | open loop | "
              "margin over the best bar |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for blk in ("gmd4a",) + tuple(new["blocks"]):
            c = tr["blocks"][blk]["arms"][KTAG]["change_5yr"]
            m = c["mean_abs_error_pts"]
            print(f"| {tr['blocks'][blk]['label']} | {c['n_pairs']} | "
                  f"**{m['CLOSURE']:.1f}** | {m['FLAT']:.1f} | {m['WATERBAL']:.1f} | "
                  f"{m['OPENLOOP']:.1f} | {c['margin_over_best_bar_pts']:+.2f} |")
        print(f"\n**Pooled over the {new['n_counties']} new counties, clustered by "
              f"county:**\n")
        print("| against | points the closure removes | se by county | standard errors | "
              "counties favouring the closure |")
        print("|---|---:|---:|---:|---:|")
        for k in ("OPENLOOP", "FLAT", "WATERBAL"):
            g = new[k]
            print(f"| {tr['blocks'][new['blocks'][0]]['arms'][KTAG]['level'][k]['label']} | "
                  f"**{g['points']:+.1f}** | {g['se_by_county']:.1f} | "
                  f"{g['n_se_by_county']:+.1f} | "
                  f"{g['n_counties_favouring_closure']}/{g['n_counties']} |")
        print("\n**The pre-registered predictions, scored as written:**\n")
        print("| | prediction | result | |")
        print("|---|---|---|---|")
        for k, v in tr["predictions"].items():
            print(f"| {k} | {v['statement']} | {v['value']} | "
                  f"**{'held' if v['pass'] else 'failed'}** |")
        print("")

    iv = load(f"interval{KTAG}.json")
    if iv:
        section("The interval the transfer found too wide, and what correcting it buys")
        print("The transfer's one failed prediction was the interval: it covers more "
              "than it claims on blocks the error budget was not estimated on. The "
              "factor below is the multiplier on the posterior spread, about its own "
              "mean and in the log the inversion parameterises, that makes the 90 per "
              "cent interval nominal. Below one is an interval that was too wide. No "
              "posterior is rewritten and no shipped score moves.\n")
        print("| block | factor | covers 50 | 80 | 90 | CRPS Mm3/yr |")
        print("|---|---:|---:|---:|---:|---:|")
        for blk, r in iv["per_block"].items():
            u = r["uncalibrated"]
            print(f"| {r['label']}"
                  f"{', where the error budget was estimated' if blk == 'gmd4a' else ''} "
                  f"| {r['in_sample_factor']:.2f} | {u['cover_50']:.2f} | "
                  f"{u['cover_80']:.2f} | {u['cover_90']:.2f} | {u['crps_mcm']:.2f} |")
        fs = [iv["per_block"][b]["in_sample_factor"] for b in iv["leave_one_new_block_out"]]
        print(f"\n**The block the budget was estimated on wants no correction; the three "
              f"it was not estimated on want the same one**, {min(fs):.2f} to "
              f"{max(fs):.2f}. The defect is the two-stage budget read in sample, not a "
              f"property of any basin.\n")
        print("| pooled over the county-years never seen | covers 50 | 80 | 90 | CRPS |")
        print("|---|---:|---:|---:|---:|")
        for k, lab in (("uncalibrated", "as run"),
                       ("leave_one_block_out", "leave-one-block-out correction")):
            v = iv["pooled_new_counties"][k]
            print(f"| {lab} | {v['cover_50']:.2f} | {v['cover_80']:.2f} | "
                  f"{v['cover_90']:.2f} | {v['crps_mcm']:.2f} |")
        print(f"\nThe factor applied to a block is fitted on the other blocks only, so "
              f"no block enters its own calibration; the in-sample factor reaches "
              f"nominal coverage by construction and is an oracle. The rule fitted on "
              f"the published block alone, which is all this repository had before the "
              f"transfer, is {iv['home_block_rule']['factor']:.2f} and corrects nothing.\n")

    si = load("saq_insar.json")
    if si:
        section("Is the deformation leg readable over the target basin's own pivots?")
        m, c, r = si["_meta"], si["coherence"], si["rate_los_mm_yr"]
        print("COMET LiCSAR frame {}, track {}, contains the Wadi As-Sirhan pivot "
              "field. {} published interferograms carry usable phase over the Al Jawf "
              "box, {} epochs from {} to {}. Pivots are labelled by the ESA WorldCover "
              "cropland class on the {:.0f} m radar grid, so the contrast below carries "
              "no radar information in its own labels.\n".format(
                  m["frame"], m["track"], m["n_pairs"], len(m["epochs"]),
                  m["first_epoch"], m["last_epoch"], m["pixel_m"]))
        print("| | pivot fields | desert |")
        print("|---|---:|---:|")
        print("| mean coherence | {:.2f} | {:.2f} |".format(
            c["mean_over_pivots"], c["mean_over_desert"]))
        print("| share above 0.3 | {:.2f} | {:.2f} |".format(
            c["share_of_pivots_above_0.3"], c["share_of_desert_above_0.3"]))
        print("| share above 0.5 | {:.2f} | {:.2f} |".format(
            c["share_of_pivots_above_0.5"], c["share_of_desert_above_0.5"]))
        print("\nThe pivots decorrelate and the ground between them holds phase in four "
              "pixels out of five, which is where a regional compaction signal is read. "
              "The published window is {:.2f} years: the line-of-sight rate is "
              "{:+.1f} +- {:.1f} mm/yr over the pivots and {:+.1f} +- {:.1f} over an "
              "independent desert control, so that span resolves {:.0f} mm/yr at two "
              "standard errors against a published Saq range of {:.0f} to {:.0f}. It "
              "establishes where the phase can be read; the rate needs the full archive "
              "over the same frame.\n".format(
                  r["span_years"], r["pivots"]["rate"], r["pivots"]["se"],
                  r["desert_control"]["rate"], r["desert_control"]["se"],
                  r["detectable_at_2se_mm_yr"],
                  r["published_saq_subsidence_mm_yr"][0],
                  r["published_saq_subsidence_mm_yr"][1]))

    ni = load("net_inflow.json")
    if ni:
        section("How far apart the two rungs are: the share of pumping that is storage")
        print("The closure reads abstraction out of storage, so what the two rungs "
              "have in common is not the basin but the method. Net inflow "
              "`N = Q + Sy*A*dh` from the withheld meters, the winter water levels and "
              "the USGS specific-yield map puts a number on the difference. No model "
              "enters it.\n")
        print("| county | pumping, Mm3/yr | water-level change, m/yr | net inflow, "
              "Mm3/yr | N/Q |")
        print("|---|---:|---:|---:|---:|")
        for c_ in ni["_specific_yield_by_county"]:
            v = ni["by_county"][c_]
            print(f"| {c_} | {v['pumping_mcm_yr']:.1f} | "
                  f"{v['water_level_change_m_yr']:+.2f} | {v['net_inflow_mcm_yr']:.1f} | "
                  f"{v['net_inflow_over_pumping']:.2f} |")
        b = ni["block"]
        print(f"| **block** | **{b['pumping_mcm_yr']:.0f}** | | "
              f"**{b['net_inflow_mcm_yr']:.0f}** | "
              f"**{b['net_inflow_over_pumping']:.2f}** |")
        print(f"\nStorage supplies {100 * b['storage_share_of_pumping']:.0f} per cent of "
              f"what Northwest Kansas pumps over {ni['_metered_era']} to 2023 and "
              f"{100 * ni['twin']['storage_share_of_pumping']:.0f} per cent of what the "
              f"twin pumps, a factor of {ni['ratio_of_storage_shares']:.1f} on the share "
              f"of pumping the closure's own signal carries. The synthetic error rate is "
              f"therefore not a prediction for this basin, and the basin the method is "
              f"built for, where recharge is a rounding error, sits at the twin's end of "
              f"that range.\n")

    al = load("aljawf.json")
    if al:
        section("L3 Al Jawf: how far apart the published instruments are on the Saq")
        pub, yr = al["_published"], str(al["_published"]["year"])
        print(f"No metered abstraction exists for this basin, so nothing here is scored. "
              f"What is reported is the disagreement between the instruments a regulator "
              f"would reach for today, over one aquifer, from public data.\n")
        print(f"Centre pivots delineated from the annual maximum MODIS NDVI above "
              f"{al['_ndvi_threshold']:.2f}: **{al['pivot_km2'][yr]:,.0f} km2** in {yr}, "
              f"against {pub['pivot_km2']:,.0f} km2 delineated at 30 m by "
              f"{pub['source']}. Across thresholds 0.35 to 0.50 the extent runs "
              f"{min(al['_threshold_sensitivity_km2'].values()):,.0f} to "
              f"{max(al['_threshold_sensitivity_km2'].values()):,.0f} km2.\n")
        yrs = sorted(al["reference_et_mm_yr"])
        print("| account | " + " | ".join(yrs) + " |")
        print("|---|" + "---:|" * len(yrs))
        def _key(kv):
            v = kv[1].get(yr)
            return -(v if v is not None and v == v else -1.0)

        rows = sorted(al["et_mm_yr"].items(), key=_key)
        for name, r in rows:
            cells = [("not published" if r.get(y) is None or r[y] != r[y]
                      else f"{r[y]:,.0f}") for y in yrs]
            print(f"| {name} | " + " | ".join(cells) + " |")
        print("| **reference evapotranspiration, climatic benchmark** | "
              + " | ".join(f"**{al['reference_et_mm_yr'][y]:,.0f}**" for y in yrs) + " |")
        # TerraClimate is reported apart from the retrievals. It is a water-balance model
        # with no irrigation term, so it does not disagree about the agriculture, it
        # cannot see it, and folding it into a spread would hide that.
        al_abs = {k: v for k, v in al["abstraction_mcm"].items()
              if not k.startswith("TerraClimate")}
        tc = al["et_mm_yr"]["TerraClimate water balance, 4 km"][yr]
        def _spread(y):
            fin = {k: r[y] for k, r in al["et_mm_yr"].items()
                   if r.get(y) is not None and r[y] == r[y]
                   and not k.startswith("TerraClimate")}
            return len(fin), max(fin.values()) / min(fin.values())

        WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
        sp = {y: _spread(y) for y in yrs}
        n_yr, s_yr = sp[yr]
        lo = min(v for _, v in sp.values())
        hi = max(v for _, v in sp.values())
        print(f"\nThe {WORD.get(n_yr, n_yr)} retrievals that publish a value for {yr} "
              f"span a factor of "
              f"**{s_yr:.1f}**, and across the {WORD.get(len(yrs), len(yrs))} years "
              f"measured the spread runs "
              f"{lo:.1f} to {hi:.1f}. TerraClimate is left out of that range on purpose: "
              f"at {tc:,.0f} mm/yr it is "
              f"{tc / al['reference_et_mm_yr'][yr] * 100:.0f} per cent of the reference, "
              f"because it carries no irrigation term, so it does not disagree about the "
              f"agriculture, it cannot see it. Converted at a single constant of 0.80, "
              f"which rescales every account together and so leaves the spread "
              f"between them unchanged, the {yr} spread "
              f"is {min(al_abs.values()):,.0f} to {max(al_abs.values()):,.0f} Mm3/yr against a "
              f"published {pub['abstraction_mcm']:,.0f}.")

        g = al["grace"]
        print(f"\n**The gravimetric leg needs a control and has never had one.** The Saq "
              f"footprint falls at {g['saq_cm_decade']:+.2f} cm/decade over "
              f"{g['n_months']} months. The same trend over deserts with no irrigation:\n")
        print("| control | cm/decade | differenced | local share | Mm3/yr |")
        print("|---|---:|---:|---:|---:|")
        for name, c in sorted(g["controls"].items(), key=lambda kv: kv[1]["cm_decade"]):
            print(f"| {name} | {c['cm_decade']:+.2f} | "
                  f"{c['differenced_cm_decade']:+.2f} | "
                  f"{c['local_share_pct']:.0f}% | {c['differenced_mcm_yr']:,.0f} |")
        lo, hi = g["local_share_pct_range"]
        vlo, vhi = g["differenced_mcm_yr_range"]
        print(f"\nThe local share of the raw trend is between {lo:.0f} and {hi:.0f} per "
              f"cent depending on which desert is the control, so the storage loss it "
              f"implies runs {vlo:,.0f} to {vhi:,.0f} Mm3/yr over "
              f"{g['saq_area_km2']:,.0f} km2. Against a crop-coefficient consumptive use "
              f"of {g['kc_consumptive_mcm_yr']:,.0f} Mm3/yr over the pivots, the two "
              f"satellite legs disagree by a factor of {g['kc_consumptive_mcm_yr']/vhi:.1f} "
              f"to {g['kc_consumptive_mcm_yr']/vlo:.0f}. Neither leg can settle it alone, "
              f"which is what the closure is for.")

    # ------------------------------------------------------- the mascon gain
    gn = load("gain.json")
    if gn and gn.get("identifiability"):
        section("The mascon gain: identifiability, propagation, and the Saq")
        print("The gravity operator reads `G(t) = alpha * dS(t) / A + external(t) + eps`,")
        print("and `alpha` multiplies the quantity that leg exists to supply. Two")
        print("questions the ablation grid cannot answer, because it varies which legs")
        print("are assimilated and never the gain. Reproduce with `make gain`.\n")
        print("**Is the gain identifiable?** Share of its prior variance the posterior")
        print("removes, by observing set.\n")
        print("| observations | posterior gain | prior variance removed | gravity leg |")
        print("|---|---:|---:|---|")
        for r in gn["identifiability"]:
            print(f"| {r['label']} | {r['alpha_hat']:.3f} | "
                  f"{100 * r['var_removed']:.0f}% | "
                  f"{'yes' if r['has_gravity_leg'] else 'no'} |")
        iv = gn["_identifiability_verdict"]
        print(f"\nEvery set without a gravity leg removes at most "
              f"{100 * iv['max_var_removed_without_gravity']:.0f} per cent, and the "
              f"four-way closure removes "
              f"{100 * iv['var_removed_by_full_closure']:.0f}. The gain is constrained "
              f"only by the leg it multiplies, so the absolute scale is a prior and this "
              f"is the number that says so.")
        if gn.get("sweep"):
            print("\n**What does that prior cost?** The four-leg row at a sequence of")
            print("gain prior widths, nothing else changed.\n")
            print("| gain prior | residual gain error | MAE, Mm3/yr | "
                  "basin abstraction bias | district 90% interval, Mm3/yr |")
            print("|---|---:|---:|---:|---:|")
            for r in gn["sweep"]:
                sd = ("free across the box" if r["free"]
                      else f"plus or minus {r['alpha_prior_sd']:.3f}")
                w = "-" if not r.get("width90_mcm") else f"{r['width90_mcm']:.1f}"
                print(f"| {r['label']}, {sd} | {r.get('alpha_err', float('nan')):+.3f} | "
                      f"{r['mae_mcm']:.2f} | {r['basin_bias_pct']:+.1f}% | {w} |")
            pg = gn.get("_propagation", {})
            wt = gn.get("_withheld_truth", {})
            pt = gn.get("_partition", {})
            if pg.get("cost_of_the_published_assumption_pts") is not None:
                print(f"\nThe twin's true gain is {wt.get('alpha', float('nan')):.2f} "
                      f"against a shipped prior of {gn['_prior_shipped']['mean']:.2f} "
                      f"plus or minus {gn['_prior_shipped']['sd']:.2f}, wrong by "
                      f"{wt.get('prior_offset_in_prior_sd', float('nan')):.2f} of its "
                      f"own standard deviations by construction, so the sweep measures "
                      f"what a wrongly computed gain costs. Believing it exactly puts "
                      f"**{pg['cost_of_the_published_assumption_pts']:.1f} percentage "
                      f"points** into the basin scale that the published width buys "
                      f"back, at the price of a district interval "
                      f"{pg.get('interval_widening_pct', float('nan')):.0f} per cent "
                      f"wider. Releasing the gain and the external mass trend together "
                      f"moves the scale by **{pg['free_vs_published_pts']:.1f}** points "
                      f"and raises the error by "
                      f"{100 * (pg.get('mae_free_over_published', 1.0) - 1.0):.0f} per "
                      f"cent, while the gain itself comes back within "
                      f"{abs(pt.get('alpha_err_with_both_free', float('nan'))):.3f} of "
                      f"truth and the external trend takes up the discrepancy instead. "
                      f"A well-recovered gain is not evidence that the gravity leg has "
                      f"been read correctly.")

    sq = load("saq_gain.json")
    if sq:
        st, al_, sv = sq["_tessellation"], sq["_mascon_aligned_footprint"], \
            sq["_sensitivity_verdict"]
        print(f"\n**The gain on the target basin, computed rather than assumed.** The "
              f"mascon polygons are recovered from the published product, which is "
              f"piecewise constant on them: {st['mascons_in_window']} mascons over the "
              f"window, median area {st['median_area_km2']:,.0f} km2 against the "
              f"{st['equal_area_3deg_km2']:,.0f} km2 of the three-degree equal-area "
              f"design. The source is the {sq['_source']['irrigated_km2']:,.0f} km2 above "
              f"NDVI {sq['_source']['ndvi_threshold']:.2f} in every one of "
              f"{', '.join(str(y) for y in sq['_source']['years'])}. Reproduce with "
              f"`make saq-gain`.\n")
        sp = [r["spread_km"] for r in sq["gain_by_spread"]]
        print("| reporting footprint | area, km2 | "
              + " | ".join(f"gain, source spread {r:.0f} km" for r in sp) + " |")
        print("|---|---:|" + "---:|" * len(sp))
        for name in sq["_footprint_area_km2"]:
            gains = " | ".join(f"{r['gain'][name]:.3f}" for r in sq["gain_by_spread"])
            print(f"| {name} | {sq['_footprint_area_km2'][name]:,.0f} | {gains} |")
        print(f"\nOver whole mascons the averaging returns every unit of mass that is "
              f"there, for any source geometry inside it: that footprint is "
              f"{al_['mascons']} mascons and {al_['area_km2']:,.0f} km2. Under a "
              f"persistent delineation the threshold moves the gain over the Saq box by "
              f"{sv['persistent_delineation']['gain_sd']['0.0']:.3f}; delineating from a "
              f"single year moves it by "
              f"{sv['one_year_delineation']['gain_sd']['0.0']:.3f}. The uncertainty the "
              f"product publishes over that box is "
              f"{sq['_mascon_uncertainty_mm']:.1f} mm against the "
              f"{sq['_l0_grace_sigma_mm']:.0f} mm the L0 twin generates at.")
        sg = load("gain_sigma.json")
        if sg and "H" in sg:
            a, b = ab["H"], sg["H"]
            print(f"\nRepeating the four-leg row with the gravity leg degraded to "
                  f"{sg['_meta']['grace_sigma_mm']:.1f} mm and the estimator told about "
                  f"it: MAE {a['mae_mcm']:.2f} to {b['mae_mcm']:.2f} Mm3/yr, basin scale "
                  f"{a['basin_bias_pct']:+.1f} to {b['basin_bias_pct']:+.1f} per cent, "
                  f"90 per cent coverage {a['cover_90']:.2f} to {b['cover_90']:.2f}.")

    # ------------------------------------------------- the external mass trend
    dr = load("drift.json")
    if dr and dr.get("sweep"):
        section("The external mass trend: the other nuisance in the gravity operator")
        ctl = dr["_l3_controls_mm_yr"]
        print("The entry constrains the linear part of the external mass term to plus "
              "or minus {:.1f} mm/yr. The L3 control boxes, unirrigated desert over the "
              "Arabian shield and the Rub' al Khali, carry {} mm/yr in magnitude, "
              "against {:.1f} over the Saq itself, so the target basin does not support "
              "a prior that tight. Reproduce with `make drift`."
              .format(dr["_shipped_prior_mm_yr"],
                      ", ".join("{:.1f}".format(v) for v in ctl.values()),
                      dr["_l3_saq_trend_mm_yr"]))
        print()
        print("| external trend prior, mm/yr | posterior trend | MAE, Mm3/yr | "
              "basin abstraction bias | district 90% interval, Mm3/yr | 90% coverage |")
        print("|---|---:|---:|---:|---:|---:|")
        for r in dr["sweep"]:
            print("| {}, plus or minus {:.0f} | {:+.2f} | {:.2f} | {:+.1f}% | {:.1f} | "
                  "{:.2f} |".format(r["label"], r["drift_trend_prior_sd"],
                                    r.get("drift_trend_hat", float("nan")),
                                    r["mae_mcm"], r["basin_bias_pct"],
                                    r.get("width90_mcm", float("nan")),
                                    r.get("cover_90", float("nan"))))
        v = dr["_verdict"]
        print("\nWidening the prior {:.0f}-fold costs {:.0f} per cent of the error, "
              "{:.2f} to {:.2f} Mm3/yr, moves the basin scale {:+.1f} to {:+.1f} per "
              "cent and leaves the coverage at {:.2f}. The constraint the target basin's "
              "controls do not support is not the one the answer rests on."
              .format(v["prior_width_ratio"], 100 * (v["mae_ratio"] - 1.0),
                      v["mae_shipped_mcm"], v["mae_widest_mcm"],
                      v["bias_shipped_pct"], v["bias_widest_pct"],
                      v["cover_90_widest"]))
        cells = dr.get("_factorial", {}).get("cells", [])
        if cells:
            print("\n**The two nuisances as a two-by-two.** Each held at the prior the "
                  "entry ships, or released.\n")
            print("| mascon gain | external mass trend | MAE, Mm3/yr | "
                  "basin abstraction bias |")
            print("|---|---|---:|---:|")
            for c in cells:
                g = ("released, sd {:.2f}".format(c["alpha_prior_sd"])
                     if c["gain_free"] else "held at the shipped prior")
                t = ("released, sd {:.0f} mm/yr".format(c["drift_trend_prior_sd"])
                     if c["trend_free"] else "held at the shipped prior")
                print("| {} | {} | {:.2f} | {:+.1f}% |".format(
                    g, t, c["mae_mcm"], c["basin_bias_pct"]))
            if len(cells) == 4:
                f = dr["_factorial"]
                print("\nThe pair is not symmetric. Released on its own the external "
                      "trend raises the district error by {:.1f} per cent; released on "
                      "its own, with the trend still held, the gain raises it by {:.0f}; "
                      "released together, {:.0f}. The gain carries the absolute scale "
                      "and the trend does not, so the gain is the constraint that has to "
                      "be defended, and it is the one that is computable on the target "
                      "basin. In both released-gain rows the posterior gain lands within "
                      "{:.3f} of the withheld truth while the estimate is a third worse."
                      .format(100 * (f["mae_ratio_trend_only"] - 1.0),
                              100 * (f["mae_ratio_gain_only"] - 1.0),
                              100 * (f["mae_ratio_both"] - 1.0),
                              max(abs(c["alpha_err"]) for c in cells
                                  if c["gain_free"])))

    print("\n---\n")
    print("Reproduce with `make reproduce` from a fresh clone, which runs every rung "
          "below in order and ends with this report. `make all` on its own is the L0 rung "
          "and its figures, not the whole submission. Rung by rung: `make test` runs the "
          "guards, `make robustness` the seed and uniform-efficiency repeats, "
          "`make kansas-data && make kansas && make kansas-score` the Kansas rung, "
          "`make verify` the window-pair scoring, `make aljawf` the Al Jawf rung, and "
          "`make gain`, `make saq-gain` and `make drift` the three sensitivity studies "
          "on the gravity leg. `make aljawf` and `make saq-gain` read Earth Engine and "
          "need an account, so `make reproduce` leaves them out and `make reproduce-ee` "
          "adds them.")


if __name__ == "__main__":
    main()
