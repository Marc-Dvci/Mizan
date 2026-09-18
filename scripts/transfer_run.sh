#!/bin/sh
# One transfer block, blind: the frozen _v3 (stage 1 + stage 2, closure and ET-only),
# then the _v5 arm. Usage: sh scripts/transfer_run.sh west [NE] [WORKERS]
B=$1; NE=${2:-250}; W=${3:-6}
PY=.venv/Scripts/python.exe
$PY scripts/11_kansas_run.py --block $B --ne $NE --na 6 --workers $W --nominal-error --rows ETH --out kansas_v3_stage1_$B.json --tag _v3s1_$B || exit 1
$PY scripts/11_kansas_run.py --block $B --ne $NE --na 6 --workers $W --budget-from kansas_posterior_ETH_v3s1_$B.npz --rows ETH,ET --out kansas_v3_$B.json --tag _v3_$B || exit 1
$PY scripts/11_kansas_run.py --block $B --ne $NE --na 6 --workers $W --config v5 --nominal-error --rows ETH --out kansas_v5_stage1_$B.json --tag _v5s1_$B || exit 1
$PY scripts/11_kansas_run.py --block $B --ne $NE --na 6 --workers $W --config v5 --budget-from kansas_posterior_ETH_v5s1_$B.npz --rows ETH --out kansas_v5_$B.json --tag _v5_$B || exit 1
echo TRANSFER RUN DONE $B
