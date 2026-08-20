#!/bin/sh
# Repeat the rows that carry the argument across independent prior ensembles, and
# re-run the headline row against a truth in which every district shares one
# consumptive fraction. Sequential, so the worker pools do not contend.
set -e
PY=.venv/Scripts/python.exe
ROWS=H,G,SAT,ET

$PY scripts/00_truth.py --eta-uniform --out truth_etauniform.npz --ws etau
$PY scripts/01_ablation.py --ne 250 --na 8 --rows H,ET --truth truth_etauniform.npz \
    --out ablation_etauniform.json --tag _etau --workers 6

for S in 6 7; do
  $PY scripts/01_ablation.py --ne 250 --na 8 --rows $ROWS --seed $S \
      --out ablation_seed$S.json --tag _s$S --workers 6
done
