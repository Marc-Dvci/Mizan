# One command reproduces every figure and table in the submission.
PY ?= .venv/Scripts/python.exe
NE ?= 250
NA ?= 8
# The uncertainty the mascon product publishes over the Saq box, measured by
# scripts/24_saq_gain.py. The twin generates its gravity leg quieter than this, so the
# four-leg row is repeated at the measured error.
SAQ_SIGMA ?= 20.4
EE_PROJECT ?= $(EARTHENGINE_PROJECT)

.PHONY: all setup env truth ablation allocation voi detection figures report test clean         robustness null kansas-data kansas kansas-score kansas-forced kansas-v5 metered-era net-inflow headline aljawf verify referee gain saq-gain drift reproduce reproduce-ee transfer-data transfer-run transfer-truth transfer reproduce-transfer interval

all: truth ablation allocation voi detection null figures report

# `all` is the L0 rung and its figures. It is not the whole submission: the Kansas
# rung, the two Earth Engine rungs and the three gravity-leg studies have their own
# targets below, and `RESULTS.md` reports whichever of them have been run. This target
# is the difference. Each step is a separate `make` so the ordering is the real one and
# the report runs last, after every rung it has to read.
reproduce:
	$(MAKE) all
	$(MAKE) robustness
	$(MAKE) kansas-data
	$(MAKE) kansas
	$(MAKE) kansas-score
	$(MAKE) verify
	$(MAKE) metered-era
	$(MAKE) net-inflow
	$(MAKE) headline
	$(MAKE) gain
	$(MAKE) drift
	$(MAKE) report
	$(MAKE) test

# The same, plus the two rungs that read Earth Engine and so need an account.
reproduce-ee:
	$(MAKE) reproduce
	$(MAKE) aljawf
	$(MAKE) saq-gain
	$(MAKE) report

# The lock is the environment the published results were produced in, so it is what
# `setup` installs. `pyproject.toml` carries version floors for anyone who wants a fresh
# resolution instead: `uv pip install -e ".[dev,ee]"`.
#
# The solver is pinned the same way the Python side is. `get_modflow` defaults to the
# latest MODFLOW-ORG/executables release, so without --release-id two clones a month
# apart can run different binaries against the same locked packages. MF6_RELEASE is the
# release the published runs used; the lock header records it beside the SHA-256 of the
# `mf6.exe` it installs, and `tests/test_guards.py` holds the three together.
MF6_RELEASE ?= 29.0

setup:
	uv venv --python 3.12 .venv
	VIRTUAL_ENV=$(PWD)/.venv uv pip install -r requirements.lock.txt
	VIRTUAL_ENV=$(PWD)/.venv uv pip install -e . --no-deps
	$(PY) -m flopy.utils.get_modflow ./bin --repo executables --release-id $(MF6_RELEASE)

# Rewrite the lock and the licence audit from the environment that is installed now.
env:
	$(PY) scripts/26_environment.py

truth:
	$(PY) scripts/00_truth.py

ablation:
	$(PY) scripts/01_ablation.py --ne $(NE) --na $(NA) --rows H,ET,A,B,C,D,E,F,G,M,HM1,HM3,SAT,HS

convergence:
	$(PY) scripts/02_convergence.py --ne $(NE) --na 9

allocation:
	$(PY) scripts/03_allocation.py

voi:
	$(PY) scripts/04_voi.py

detection:
	$(PY) scripts/00_truth.py --hidden 3:40 --out truth_hidden.npz
	$(PY) scripts/06_detection.py --ne $(NE) --na $(NA)

figures:
	$(PY) scripts/05_figures.py

# Every generated number, tabulated. This file is committed, so a reader can check a
# number against the code without running anything.
report:
	$(PY) scripts/07_report.py > RESULTS.md

test:
	$(PY) -m pytest tests -q

robustness:
	sh scripts/robustness.sh

# The published Kansas configuration. Override to score another one:
#   make verify KTAG=_v4
KTAG ?= _v3

kansas-data:
	$(PY) scripts/10_kansas_fetch.py --workers 4

null:
	$(PY) scripts/13_null.py

# Two stages. The first is weighted at instrument error and supplies the converged
# residual the error budget is read from; a budget read off an unconverged residual
# measures the distance the ensemble has not travelled yet, not the distance it cannot.
#
# `_v3` is the published configuration: per-site error budget, layer base taken from the
# USGS saturated-thickness field rather than estimated. `_v3p` is the pooled-budget
# sensitivity, reported beside it. Both stay runnable.
kansas:
	$(PY) scripts/11_kansas_run.py --ne $(NE) --na 6 --workers 6 --nominal-error --rows ETH --out kansas_v3_stage1.json --tag _v3s1
	$(PY) scripts/11_kansas_run.py --ne $(NE) --na 6 --workers 6 --budget-from kansas_posterior_ETH_v3s1.npz --out kansas_v3.json --tag _v3
	$(PY) scripts/11_kansas_run.py --ne $(NE) --na 6 --workers 6 --pooled-error --budget-from kansas_posterior_ETH_v3s1.npz --out kansas_v3p.json --tag _v3p
	$(PY) scripts/16_kansas_convergence.py --ne 80 --workers 6

# The precipitation-forced recharge, `_v4`, run and rejected in the decision log. It
# is a driver flag so that `make kansas` reproduces the published `_v3`.
kansas-forced:
	$(PY) scripts/11_kansas_run.py --ne $(NE) --na 6 --workers 6 --forced-recharge --nominal-error --rows ETH --out kansas_v4_stage1.json --tag _v4s1
	$(PY) scripts/11_kansas_run.py --ne $(NE) --na 6 --workers 6 --forced-recharge --budget-from kansas_posterior_ETH_v4s1.npz --out kansas_v4.json --tag _v4

# `_v5`: specific yield on the USGS map times one multiplier, the conductivity prior
# centred on the USGS map, the deep-percolation share a parameter. The prediction is
# written in the decision log before the run.
kansas-v5:
	$(PY) scripts/11_kansas_run.py --ne $(NE) --na 6 --workers 6 --config v5 --nominal-error --rows ETH --out kansas_v5_stage1.json --tag _v5s1
	$(PY) scripts/11_kansas_run.py --ne $(NE) --na 6 --workers 6 --config v5 --budget-from kansas_posterior_ETH_v5s1.npz --out kansas_v5.json --tag _v5

kansas-score: null
	$(PY) scripts/12_kansas_anomaly.py --tag $(KTAG)
	$(PY) scripts/14_kansas_resolution.py --tag $(KTAG) --out kansas_resolution$(KTAG).json
	$(PY) scripts/15_kansas_shrink.py --tag $(KTAG)

# The blind transfer. `transfer-data` fetches a block's estimator inputs and refuses
# its use files; `transfer-run` runs the frozen `_v3` and the `_v5` arm on it, unscored;
# `transfer-truth` fetches the use files, which the fetcher allows only once the
# block's predictions are in a committed DECISION_LOG.md; `transfer` scores every
# block that has both. BLOCK is one of west, gmd3w, gmd3e.
BLOCK ?= west
TROWS ?= ETH,ET

transfer-data:
	$(PY) scripts/10_kansas_fetch.py --block $(BLOCK) --workers 4

transfer-run:
	$(PY) scripts/11_kansas_run.py --block $(BLOCK) --ne $(NE) --na 6 --workers 6 --nominal-error --rows ETH --out kansas_v3_stage1_$(BLOCK).json --tag _v3s1_$(BLOCK)
	$(PY) scripts/11_kansas_run.py --block $(BLOCK) --ne $(NE) --na 6 --workers 6 --budget-from kansas_posterior_ETH_v3s1_$(BLOCK).npz --rows $(TROWS) --out kansas_v3_$(BLOCK).json --tag _v3_$(BLOCK)
	$(PY) scripts/11_kansas_run.py --block $(BLOCK) --ne $(NE) --na 6 --workers 6 --config v5 --nominal-error --rows ETH --out kansas_v5_stage1_$(BLOCK).json --tag _v5s1_$(BLOCK)
	$(PY) scripts/11_kansas_run.py --block $(BLOCK) --ne $(NE) --na 6 --workers 6 --config v5 --budget-from kansas_posterior_ETH_v5s1_$(BLOCK).npz --rows ETH --out kansas_v5_$(BLOCK).json --tag _v5_$(BLOCK)

transfer-truth:
	$(PY) scripts/10_kansas_fetch.py --block $(BLOCK) --what wuse

transfer:
	$(PY) scripts/30_transfer.py --tag _v3 --arms _v5

# Every transfer block end to end, in the order the protocol requires. The use files are
# fetched last and the fetcher itself refuses them until each block's predictions are in
# a committed decision log, so this target reproduces the protocol and not only the
# numbers. It is separate from `reproduce` because it is three more inversions.
reproduce-transfer:
	for b in west gmd3w gmd3e; do 	  $(MAKE) transfer-data BLOCK=$$b; 	  sh scripts/transfer_run.sh $$b $(NE) 6; 	  $(MAKE) transfer-truth BLOCK=$$b; 	done
	$(MAKE) transfer
	$(MAKE) interval
	$(MAKE) report

# The interval the transfer found too wide, measured per block and corrected
# leave-one-block-out, so the factor a block is scored under was fitted without it.
interval:
	$(PY) scripts/31_interval.py --tag $(KTAG)

# Every Kansas score again, on the years the truth is fully metered. WIMAS codes each
# report by how it was measured; the block is meter-coded from 2009. The truth is also
# rebuilt from the per-point use file, which the history page under-counts.
metered-era:
	$(PY) scripts/27_metered_era.py --tag $(KTAG)

# How much of what Kansas pumps the aquifer replaces within the year, against the twin,
# which is the size of the difference between the two rungs.
net-inflow:
	$(PY) scripts/28_net_inflow.py

# The two comparisons the entry rests on, each with the error bar its claim needs:
# clustered by county for a transfer claim, by year for a within-block one.
headline:
	$(PY) scripts/29_headline.py --tag $(KTAG)

# L3. Everything is read live from Earth Engine, so this target needs an authenticated
# project and nothing else: `earthengine authenticate`, then set EARTHENGINE_PROJECT to
# your own cloud project. Registration is free and every asset read here is public.
# Nothing in this rung is fitted and nothing is scored: it reports how far the published
# instruments are from each other over one basin.
aljawf:
	$(PY) scripts/20_aljawf.py
	$(PY) scripts/21_aljawf_figure.py

clean:
	rm -rf runs/ens runs/alloc runs/jac

# The verification rung: what a reduction target actually asks, scored on the meters.
verify:
	$(PY) scripts/18_ladder.py --tag $(KTAG)
	$(PY) scripts/19_verify.py --tag $(KTAG)
	$(PY) scripts/22_verify_figure.py --tag $(KTAG)
# The same four panels on a two-by-two grid. The banner is a page-width figure for
# the documents; at 16:9 it shrinks every panel past reading size, so the deck and
# the film use this one.
	$(PY) scripts/22_verify_figure.py --tag $(KTAG) --layout grid --out fig12_verify_deck.png

referee:
	$(PY) scripts/17_referee.py --ne 100 --na 3 --workers 6 --tag $(KTAG)

# Is the mascon gain identifiable, and what does its uncertainty cost the absolute
# scale? Five runs of the four-leg row differing only in the width of the gain prior,
# plus the configuration the decision log records as rejected.
gain:
	$(PY) scripts/01_ablation.py --ne $(NE) --na $(NA) --rows H --alpha-sd 0.001 --out gain_fixed.json --tag _gfix
	$(PY) scripts/01_ablation.py --ne $(NE) --na $(NA) --rows H --alpha-sd 0.02  --out gain_tight.json --tag _gtight
	$(PY) scripts/01_ablation.py --ne $(NE) --na $(NA) --rows H --alpha-sd 0.04  --out gain_pub.json   --tag _gpub
	$(PY) scripts/01_ablation.py --ne $(NE) --na $(NA) --rows H --alpha-sd 0.08  --out gain_loose.json --tag _gloose
	$(PY) scripts/01_ablation.py --ne $(NE) --na $(NA) --rows H --alpha-sd 0.5 --drift-free --out gain_free.json --tag _gfree
	$(PY) scripts/01_ablation.py --ne $(NE) --na $(NA) --rows H --grace-sigma $(SAQ_SIGMA) --out gain_sigma.json --tag _gsig
	$(PY) scripts/23_gain.py

# The other half of the degenerate pair. The external mass trend is constrained to plus
# or minus 1.0 mm/yr, and the L3 controls over unirrigated desert measure several times
# that, so the constraint is put on its own axis rather than argued. The third run is the
# fourth cell of the two-by-two: the gain released with the trend held, which is the
# cell `make gain` does not produce.
drift:
	$(PY) scripts/01_ablation.py --ne $(NE) --na $(NA) --rows H --drift-sd 4.0  --out drift_wide.json    --tag _dwide
	$(PY) scripts/01_ablation.py --ne $(NE) --na $(NA) --rows H --drift-sd 10.0 --out drift_control.json --tag _dctl
	$(PY) scripts/01_ablation.py --ne $(NE) --na $(NA) --rows H --alpha-sd 0.5 --drift-sd 1.0 --out gain_alphafree.json --tag _gafree
	$(PY) scripts/25_drift.py

# What the gain is on the target basin, from the mascon geometry the product itself
# carries. Reads live from Earth Engine, so it needs an authenticated project. Pass
# --reuse to redraw the figure from the cached reads instead of querying again.
saq-gain:
	$(PY) scripts/24_saq_gain.py --project $(EE_PROJECT)
