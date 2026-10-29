#!/bin/bash
# Hourly health check for the ELR study. Prints a compact report; no side effects.
# HOST: ssh alias of the GPU host. JUMP (optional): ssh alias of a fallback jump host.
HOST=${HOST:?set HOST to the GPU host ssh alias}
CMD="${PY:-python} ${EXP:-/data/exp_elrshard}/code/study/watch_remote.py"
ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" "$CMD" && exit 0
[ -n "${JUMP:-}" ] && ssh -o BatchMode=yes -o ConnectTimeout=15 "$JUMP" \
  "ssh -o BatchMode=yes -o ConnectTimeout=15 $HOST '$CMD'"
