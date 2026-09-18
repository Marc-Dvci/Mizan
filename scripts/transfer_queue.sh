#!/bin/sh
# Run the remaining transfer blocks one after another, each once its inputs are on
# disk and the previous block's run has finished. Usage: sh scripts/transfer_queue.sh
wait_for() { until [ -f "$1" ]; do sleep 30; done; }
wait_done() { until grep -q "TRANSFER RUN DONE $1" runs/transfer_$1.log 2>/dev/null; do sleep 30; done; }
wait_done west
for B in gmd3w gmd3e; do
  wait_for data/kansas/precip_annual_$B.json
  wait_for data/kansas/wizard_levels_$B.json
  sh scripts/transfer_run.sh $B 250 6 > runs/transfer_$B.log 2> runs/transfer_$B.log.err
done
echo QUEUE DONE
