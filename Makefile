# One command reproduces every figure and table in the submission.
PY ?= .venv/Scripts/python.exe
NE ?= 250
NA ?= 8

.PHONY: all setup truth ablation allocation voi detection figures report test clean         robustness null kansas-data kansas kansas-score aljawf verify referee

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

KTAG ?= _v4

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
	$(PY) scripts/12_kansas_anomaly.py --tag _v3
	$(PY) scripts/14_kansas_resolution.py --tag _v3 --out kansas_resolution_v3.json
	$(PY) scripts/15_kansas_shrink.py --tag _v3

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
