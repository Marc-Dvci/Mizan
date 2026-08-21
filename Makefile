# One command reproduces every figure and table in the submission.
PY ?= .venv/Scripts/python.exe
NE ?= 250
NA ?= 8
# The uncertainty the mascon product publishes over the Saq box, measured by
# scripts/24_saq_gain.py. The twin generates its gravity leg quieter than this, so the
# four-leg row is repeated at the measured error.
SAQ_SIGMA ?= 20.4
EE_PROJECT ?= $(EARTHENGINE_PROJECT)

.PHONY: all setup truth ablation allocation voi detection figures report test clean         robustness null kansas-data kansas kansas-score aljawf verify referee gain saq-gain drift

all: truth ablation allocation voi detection null figures report

setup:
	uv venv --python 3.12 .venv
	VIRTUAL_ENV=$(PWD)/.venv uv pip install -e ".[dev]"
	$(PY) -m flopy.utils.get_modflow ./bin --repo executables

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

report:
	$(PY) scripts/07_report.py > ../09_RESULTS.md

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

kansas-score: null
	$(PY) scripts/12_kansas_anomaly.py --tag $(KTAG)
	$(PY) scripts/14_kansas_resolution.py --tag $(KTAG) --out kansas_resolution$(KTAG).json
	$(PY) scripts/15_kansas_shrink.py --tag $(KTAG)

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
