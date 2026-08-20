# One command reproduces every figure and table in the submission.
PY ?= .venv/Scripts/python.exe
NE ?= 250
NA ?= 8

.PHONY: all setup truth ablation allocation voi detection figures report test clean         robustness kansas-data kansas

all: truth ablation allocation voi detection figures report

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

kansas-data:
	$(PY) scripts/10_kansas_fetch.py --workers 4

kansas:
	$(PY) scripts/11_kansas_run.py --ne $(NE) --na $(NA) --workers 6

clean:
	rm -rf runs/ens runs/alloc runs/jac
