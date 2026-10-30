#!/usr/bin/env bash
# Phase 2b: confirm the new nprobe-256 results with repeats, and redo the 10M ablation with the
# data pre-loaded (the first row of phase2 B paid 52,000 page faults loading the base file).
# One password prompt, then unattended. Roughly 10-15 minutes.
#
#  E. 10M ablation again, base file read into the page cache first
#  F. 100M cold, three settings x 3 runs, alternating: 128/5000/50, 256/5000/50, 256/10000/50
#  G. one exploratory step further: nprobe 512
set -euo pipefail
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

DATA=${DATA:-$HOME/binvec-data/deep}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$HOME/binvec-target}
export RUSTFLAGS=${RUSTFLAGS:-"-C target-cpu=native"}
BIN=$CARGO_TARGET_DIR/release/binvec
HOST=$(uname -n)
Q="$DATA/query.public.10K.fbin"

I10="$DATA/idx-10M-4096-rot-unbiased.bin";    F10="$DATA/idx-10M-4096-fastscan.bin"
R10="$DATA/refine-10M-4096.bin"
B10="$DATA/base.10M.fbin";                     G10="$DATA/gt-10M.bin"
F100="$DATA/idx-100M-16384-fastscan.bin"
R100="$DATA/refine-100M-16384.bin"
B100="$DATA/base.100M.fbin";                   G100="$DATA/gt-100M.bin"
HEAD="nprobe refine rerank recall@10   p50 ms   p99 ms  mean ms     QPS  |   cent   scan refine rerank |   majflt  swapin swapout"

cargo build --release
mkdir -p results
drop_caches() { sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
sudo -v
while true; do sudo -n true; sleep 60; done 2>/dev/null &
KEEPALIVE=$!
trap 'kill $KEEPALIVE 2>/dev/null' EXIT
stamp() { echo; echo "######## $* ($(date +%H:%M))"; }

stamp "E. 10M in RAM, base pre-loaded: layout x collector"
cat "$B10" "$R10" "$I10" "$F10" > /dev/null
for v in "bytes heap" "bytes reservoir" "fastscan heap" "fastscan reservoir"; do
  set -- $v
  idx=$I10; [ "$1" = fastscan ] && idx=$F10
  flag=(); [ "$2" = heap ] && flag=(--heap)
  echo "-- $1 + $2"
  $BIN bench --index "$idx" --base "$B10" --queries "$Q" --gt "$G10" --refine "$R10" \
    --nprobe 64,128 --refine-k 2000,5000 --rerank 50 --nq-latency 1000 "${flag[@]}" \
    --csv results/phase2b-10M.csv --label "deep10M $HOST v0.5 $1+$2 preloaded" 2>/dev/null
done

COMMON=(--index "$F100" --base "$B100" --queries "$Q" --gt "$G100" --refine "$R100"
        --warmup 0 --prefetch --nq-latency 500)
run() {  # args: nprobe refine_k rerank csv label [extra flags]
  local np=$1 rk=$2 rr=$3 csv=$4 label=$5; shift 5
  drop_caches
  $BIN bench "${COMMON[@]}" --nprobe "$np" --refine-k "$rk" --rerank "$rr" "$@" \
    --csv "$csv" --label "$label" 2>/dev/null | tail -1
}

stamp "F. 100M cold, 3 runs each, alternating"
echo "$HEAD"
for rep in 1 2 3; do
  for s in "128 5000 50" "256 5000 50" "256 10000 50"; do
    set -- $s
    run $1 $2 $3 results/phase2b-repeats.csv "deep100M $HOST v0.5 cold rep$rep" --nq 2000
  done
done

stamp "G. 100M cold, nprobe 512"
echo "$HEAD"
run 512 10000 50 results/phase2b-repeats.csv "deep100M $HOST v0.5 cold explore" --nq 2000
run 512 20000 100 results/phase2b-repeats.csv "deep100M $HOST v0.5 cold explore" --nq 2000
$BIN ceiling --index "$F100" --queries "$Q" --gt "$G100" --nprobe 256,384,512 2>/dev/null

free -m
stamp "phase 2b done"
