#!/usr/bin/env bash
# Phase 3b, Deep 100M cold: re-check the planner (auto) with page-fault and swap counters.
# In phase 3, auto took the same plans as scan at 50/10/1% yet its refine stage was 10-40x
# slower, and the slowdown grew within the run. Same code path, so: memory, not the algorithm?
# This run alternates auto and scan in separate cold processes and records faults and swap.
# Also: the inverted-index path with deeper prefetching, on the two rare labels.
# One password prompt, then unattended. Roughly 15-20 minutes.
set -euo pipefail
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

DATA=${DATA:-$HOME/binvec-data/deep}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$HOME/binvec-target}
export RUSTFLAGS=${RUSTFLAGS:-"-C target-cpu=native"}
BIN=$CARGO_TARGET_DIR/release/binvec
HOST=$(uname -n)
Q="$DATA/query.public.10K.fbin"
F100="$DATA/idx-100M-16384-fastscan.bin"; R100="$DATA/refine-100M-16384.bin"; B100="$DATA/base.100M.fbin"

cargo build --release
mkdir -p results
drop_caches() { sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
sudo -v
while true; do sudo -n true; sleep 60; done 2>/dev/null &
KEEPALIVE=$!
trap 'kill $KEEPALIVE 2>/dev/null' EXIT
stamp() { echo; echo "######## $* ($(date +%H:%M))"; }

run() {  # args: kind mode label [extra flags]
  local kind=$1 m=$2 label=$3; shift 3
  drop_caches
  free -m | awk 'NR==2 {printf "   [mem before: used %s MB, free %s MB, cache %s MB]\n", $3, $4, $6}'
  $BIN fbench --index "$F100" --base "$B100" --refine "$R100" --queries "$Q" \
    --labels "$DATA/labels-100M-$kind.bin" --gt "$DATA/gt-100M-filtered-$kind.bin" \
    --mode $m --nprobe 256 --refine-k 5000 --rerank 50 --budget 20000 --nq-latency 100 \
    --prefetch --csv results/phase3b-100M-cold.csv --label "deep100M $HOST v0.6 $kind cold $label" \
    "$@" 2>/dev/null
}

for kind in uniform correlated; do
  stamp "$kind: auto, scan, auto (alternating, each cold)"
  run $kind auto rep1
  run $kind scan rep1
  run $kind auto rep2
  stamp "$kind: inverted index on the rare labels, deeper prefetch"
  run $kind postings prefetch16 --skip-postings-above 200000
done

free -m
stamp "phase 3b done"
