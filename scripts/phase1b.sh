#!/usr/bin/env bash
# Phase 1b, Deep 100M: the refine tier with memory-mapped index files.
# One password prompt, then unattended. Roughly 25-35 minutes.
#
#  G. old vs new loading, same setting: --copy-load (v0.3: arrays read into owned memory)
#     vs memory-mapped (v0.4). Watch the swap and major-fault columns.
#  H. cold-cache grid with the refine tier, memory-mapped
#  I. the two headline candidates, 3 runs each, alternating so run order can't bias them
set -euo pipefail
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

DATA=${DATA:-$HOME/binvec-data/deep}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$HOME/binvec-target}
export RUSTFLAGS=${RUSTFLAGS:-"-C target-cpu=native"}
BIN=$CARGO_TARGET_DIR/release/binvec
HOST=$(uname -n)
I100="$DATA/idx-100M-16384-rot-unbiased.bin"
R100="$DATA/refine-100M-16384.bin"
COMMON=(--index "$I100" --base "$DATA/base.100M.fbin" --queries "$DATA/query.public.10K.fbin"
        --gt "$DATA/gt-100M.bin" --refine "$R100" --warmup 0 --prefetch --nq-latency 500)
HEAD="nprobe refine rerank recall@10   p50 ms   p99 ms  mean ms     QPS  |   cent   scan refine rerank |   majflt  swapin swapout"

cargo build --release
mkdir -p results
drop_caches() { sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
sudo -v
while true; do sudo -n true; sleep 60; done 2>/dev/null &
KEEPALIVE=$!
trap 'kill $KEEPALIVE 2>/dev/null' EXIT
stamp() { echo; echo "######## $* ($(date +%H:%M))"; }
run() {  # args: nprobe refine_k rerank csv label [extra flags]
  local np=$1 rk=$2 rr=$3 csv=$4 label=$5; shift 5
  drop_caches
  $BIN bench "${COMMON[@]}" --nprobe "$np" --refine-k "$rk" --rerank "$rr" "$@" \
    --csv "$csv" --label "$label" 2>/dev/null | tail -1
}

free -m

stamp "G. loading: copy (v0.3) vs memory-mapped (v0.4), nprobe 128 / refine 5000 / rerank 50"
echo "$HEAD"
echo "-- copy-load"
run 128 5000 50 results/phase1b-load.csv "deep100M $HOST v0.4 copy-load" --nq 2000 --copy-load
echo "-- memory-mapped"
run 128 5000 50 results/phase1b-load.csv "deep100M $HOST v0.4 mmap" --nq 2000

stamp "H. cold-cache grid, refine tier, memory-mapped"
echo "$HEAD"
for np in 64 128; do
  for rk in 2000 5000; do
    for rr in 20 50; do
      run $np $rk $rr results/phase1b-grid.csv "deep100M $HOST v0.4 cold" --nq 5000
    done
  done
done

stamp "I. headline candidates, 3 runs each, alternating"
echo "$HEAD"
for rep in 1 2 3; do
  run 64 2000 20 results/phase1b-repeats.csv "deep100M $HOST v0.4 cold rep$rep" --nq 2000
  run 128 5000 50 results/phase1b-repeats.csv "deep100M $HOST v0.4 cold rep$rep" --nq 2000
done

free -m
stamp "phase 1b done"
