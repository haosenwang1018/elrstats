#!/bin/bash
# Launch the ELR measurement study, one GPU per run, through the `expq` job queue.
# Usage (on the GPU host): EXP=/data/exp_elrshard PY=python bash study/launch.sh STEPS [batch2|batch3]
set -euo pipefail
STEPS=${1:-15000}
EXP=${EXP:-/data/exp_elrshard}
PY=${PY:-python}
run() {  # label, args...
  local label=$1; shift
  expq $EXP 1 "study-$label" -- bash -c "cd $EXP/code && $PY study/train.py $* --steps $STEPS --out $EXP/runs/$label > $EXP/runs_$label.log 2>&1"
}
mkdir -p $EXP/runs
if [ -z "${2:-}" ]; then  # batch 1 only when no batch flag is given
run adamw-lr6e-4-wd0.1-lin   --opt adamw --lr 6e-4  --wd 0.1 --sched linear
run adamw-lr6e-4-wd0.1-const --opt adamw --lr 6e-4  --wd 0.1 --sched const
run adamw-lr2.4e-3-wd0.1-lin --opt adamw --lr 2.4e-3 --wd 0.1 --sched linear
run adamw-lr6e-4-wd0-lin     --opt adamw --lr 6e-4  --wd 0   --sched linear
run muon-lr6e-4-wd0.1-lin    --opt muon  --lr 6e-4  --wd 0.1 --sched linear
run muon-lr6e-4-wd0.1-const  --opt muon  --lr 6e-4  --wd 0.1 --sched const
run muon-lr2.4e-3-wd0.1-lin  --opt muon  --lr 2.4e-3 --wd 0.1 --sched linear
run muon-lr6e-4-wd0-lin      --opt muon  --lr 6e-4  --wd 0   --sched linear
fi

# Batch 2 (added after design review, 2026-10-29): pure implicit decay (const, no WD)
# and a const high-lr pair for the lr-compression question.
if [ "${2:-}" = "batch2" ]; then
  run adamw-lr6e-4-wd0-const    --opt adamw --lr 6e-4   --wd 0   --sched const
  run muon-lr6e-4-wd0-const     --opt muon  --lr 6e-4   --wd 0   --sched const
  run adamw-lr2.4e-3-wd0.1-const --opt adamw --lr 2.4e-3 --wd 0.1 --sched const
  run muon-lr2.4e-3-wd0.1-const  --opt muon  --lr 2.4e-3 --wd 0.1 --sched const
fi

# Batch 3 (2026-10-29): seed 1 of four core configs, to measure seed noise.
if [ "${2:-}" = "batch3" ]; then
  run adamw-lr6e-4-wd0.1-lin-s1   --opt adamw --lr 6e-4 --wd 0.1 --sched linear --seed 1
  run adamw-lr6e-4-wd0.1-const-s1 --opt adamw --lr 6e-4 --wd 0.1 --sched const  --seed 1
  run muon-lr6e-4-wd0.1-lin-s1    --opt muon  --lr 6e-4 --wd 0.1 --sched linear --seed 1
  run muon-lr6e-4-wd0.1-const-s1  --opt muon  --lr 6e-4 --wd 0.1 --sched const  --seed 1
fi
