#!/usr/bin/env bash
# Everything that follows the main ablation grid, ordered so the decisive results land
# first and the extra ablation rows last.
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
$PY -u scripts/03_allocation.py --members 24 --workers 6
$PY -u scripts/04_voi.py --nmeters 20
$PY -u scripts/00_truth.py --hidden 3:40 --out truth_hidden.npz
$PY -u scripts/06_detection.py --ne 250 --na 8
$PY -u scripts/01_ablation.py --ne 250 --na 8 --rows SAT,HS
$PY -u scripts/05_figures.py
$PY -u scripts/07_report.py > ../09_RESULTS.md
