#!/usr/bin/env bash
# Phase 1 experiments, Deep 10M then Deep 100M. One password prompt, then unattended.
# Roughly 1.5-2 hours. Keep the laptop plugged in and set not to sleep.
#
#  10M:  A. build the 4-bit refine tier, benchmark with and without it (data in RAM)
#        B. exact vs two-level (approximate) centroid assignment: build time and ceiling
# 100M:  C. build the refine tier (one pass over 38 GB)
#        D. cold-cache benchmark with the refine tier
#        E. per-stage timing for nprobe 64 vs 128 (the unexplained latency result), alternating
#        F. two-level build at 100M: build time and ceiling vs the existing exact index
set -euo pipefail
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

DATA=${DATA:-$HOME/binvec-data/deep}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$HOME/binvec-target}
export RUSTFLAGS=${RUSTFLAGS:-"-C target-cpu=native"}
BIN=$CARGO_TARGET_DIR/release/binvec
Q="$DATA/query.public.10K.fbin"
HOST=$(uname -n)

cargo build --release
mkdir -p results
drop_caches() { sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
sudo -v
while true; do sudo -n true; sleep 60; done 2>/dev/null &
KEEPALIVE=$!
trap 'kill $KEEPALIVE 2>/dev/null' EXIT
stamp() { echo; echo "######## $* ($(date +%H:%M))"; }

# ---------------------------------------------------------------- 10M
I10="$DATA/idx-10M-4096-rot-unbiased.bin"
R10="$DATA/refine-10M-4096.bin"
B10="$DATA/base.10M.fbin"; G10="$DATA/gt-10M.bin"

stamp "A1. 10M refine tier build"
[ -s "$R10" ] || $BIN refine --index "$I10" --base "$B10" --out "$R10"

stamp "A2. 10M baseline, no refine tier (in RAM)"
$BIN bench --index "$I10" --base "$B10" --queries "$Q" --gt "$G10" \
  --nprobe 32,64,128 --rerank 100,500 --nq-latency 1000 \
  --csv results/phase1-10M.csv --label "deep10M $HOST v0.4" 2>/dev/null

stamp "A3. 10M with refine tier"
$BIN bench --index "$I10" --base "$B10" --queries "$Q" --gt "$G10" --refine "$R10" \
  --nprobe 32,64,128 --refine-k 1000,2000,5000 --rerank 10,20,50,100 --nq-latency 1000 \
  --csv results/phase1-10M.csv --label "deep10M $HOST v0.4" 2>/dev/null

stamp "B. 10M exact vs two-level assignment"
for ap in 0 8; do
  echo "-- assign_probe $ap"
  $BIN build --base "$B10" --nlist 4096 --train-n 500000 --assign-probe $ap \
    --out "$DATA/idx-10M-4096-ap$ap.bin" 2>&1 | grep -E "agrees|k-means done|assigned|total"
  $BIN ceiling --index "$DATA/idx-10M-4096-ap$ap.bin" --queries "$Q" --gt "$G10" \
    --nprobe 16,32,64,128 2>/dev/null
done

# ---------------------------------------------------------------- 100M
I100="$DATA/idx-100M-16384-rot-unbiased.bin"
R100="$DATA/refine-100M-16384.bin"
B100="$DATA/base.100M.fbin"; G100="$DATA/gt-100M.bin"

stamp "C. 100M refine tier build (one pass over 38 GB)"
[ -s "$R100" ] || $BIN refine --index "$I100" --base "$B100" --out "$R100"

stamp "D. 100M cold cache, refine tier, prefetch (each setting its own process)"
echo "nprobe refine rerank recall@10   p50 ms   p99 ms  mean ms     QPS  |   cent   scan refine rerank"
for np in 64 128; do
  for rk in 2000 5000; do
    for rr in 20 50 100; do
      drop_caches
      $BIN bench --index "$I100" --base "$B100" --queries "$Q" --gt "$G100" --refine "$R100" \
        --nprobe $np --refine-k $rk --rerank $rr --nq-latency 500 --nq 5000 --warmup 0 --prefetch \
        --csv results/phase1-100M-cold.csv --label "deep100M $HOST v0.4 cold" 2>/dev/null | tail -1
    done
  done
done

stamp "E. 100M per-stage timing, no refine tier, rerank 500, nprobe 64 vs 128 alternating"
echo "nprobe refine rerank recall@10   p50 ms   p99 ms  mean ms     QPS  |   cent   scan refine rerank"
for rep in 1 2 3; do
  for np in 64 128; do
    drop_caches
    $BIN bench --index "$I100" --base "$B100" --queries "$Q" --gt "$G100" \
      --nprobe $np --rerank 500 --nq-latency 500 --nq 2000 --warmup 0 --prefetch \
      --csv results/phase1-100M-stages.csv --label "deep100M $HOST v0.4 cold rep$rep" 2>/dev/null | tail -1
  done
done

stamp "F. 100M two-level build (assign_probe 8) vs the existing exact index"
F100="$DATA/idx-100M-16384-ap8.bin"
[ -s "$F100" ] || $BIN build --base "$B100" --nlist 16384 --train-n 1000000 --assign-probe 8 --out "$F100" \
  2>&1 | grep -E "agrees|k-means done|assigned|encoded|total" | tee results/build-100M-ap8.log
echo "-- ceiling, exact index (built in 3,174 s)"
$BIN ceiling --index "$I100" --queries "$Q" --gt "$G100" --nprobe 32,64,128 2>/dev/null
echo "-- ceiling, two-level index"
$BIN ceiling --index "$F100" --queries "$Q" --gt "$G100" --nprobe 32,64,128 2>/dev/null

stamp "phase 1 done"
