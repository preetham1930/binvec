#!/usr/bin/env bash
# Phase 2: a faster scan. FastScan SIMD kernel, reservoir candidate collector, and a vectorised,
# slot-ordered 4-bit refine stage. One password prompt, then unattended. Roughly 40-50 minutes.
#
#  A. convert the existing 10M and 100M indexes to the FastScan layout (no rebuild; the
#     refine files stay valid because vector order doesn't change)
#  B. 10M in RAM: 2x2 ablation, code layout (bytes / fastscan) x collector (heap / reservoir)
#  C. 100M cold: old path (bytes + heap) vs new path (fastscan + reservoir) on the two headline
#     settings, 3 runs each, alternating so run order can't bias them
#  D. 100M cold grid on the new path, including nprobe 256 (affordable now the scan is cheaper)
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
I100="$DATA/idx-100M-16384-rot-unbiased.bin"; F100="$DATA/idx-100M-16384-fastscan.bin"
R100="$DATA/refine-100M-16384.bin"
B100="$DATA/base.100M.fbin";                   G100="$DATA/gt-100M.bin"
HEAD="nprobe refine rerank recall@10   p50 ms   p99 ms  mean ms     QPS  |   cent   scan refine rerank |   majflt  swapin swapout"

cargo build --release
cargo test --release -q 2>&1 | grep "test result" | head -1
mkdir -p results
drop_caches() { sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
sudo -v
while true; do sudo -n true; sleep 60; done 2>/dev/null &
KEEPALIVE=$!
trap 'kill $KEEPALIVE 2>/dev/null' EXIT
stamp() { echo; echo "######## $* ($(date +%H:%M))"; }
grep -q avx2 /proc/cpuinfo && echo "CPU has AVX2: SIMD kernel in use" || echo "no AVX2: scalar fallback"
free -m

stamp "A. convert to the FastScan layout"
[ -s "$F10" ]  || $BIN convert --index "$I10"  --out "$F10"
[ -s "$F100" ] || $BIN convert --index "$I100" --out "$F100"
ls -l "$I10" "$F10" "$I100" "$F100"

stamp "B. 10M in RAM: layout x collector"
for v in "bytes heap" "bytes reservoir" "fastscan heap" "fastscan reservoir"; do
  set -- $v
  idx=$I10; [ "$1" = fastscan ] && idx=$F10
  flag=(); [ "$2" = heap ] && flag=(--heap)
  echo "-- $1 + $2"
  $BIN bench --index "$idx" --base "$B10" --queries "$Q" --gt "$G10" --refine "$R10" \
    --nprobe 64,128 --refine-k 2000,5000 --rerank 50 --nq-latency 1000 "${flag[@]}" \
    --csv results/phase2-10M.csv --label "deep10M $HOST v0.5 $1+$2" 2>/dev/null
done

COMMON=(--base "$B100" --queries "$Q" --gt "$G100" --refine "$R100"
        --warmup 0 --prefetch --nq-latency 500)
run() {  # args: index nprobe refine_k rerank csv label [extra flags]
  local idx=$1 np=$2 rk=$3 rr=$4 csv=$5 label=$6; shift 6
  drop_caches
  $BIN bench --index "$idx" "${COMMON[@]}" --nprobe "$np" --refine-k "$rk" --rerank "$rr" "$@" \
    --csv "$csv" --label "$label" 2>/dev/null | tail -1
}

stamp "C. 100M cold: old path vs new path, 3 runs each, alternating"
echo "$HEAD"
for rep in 1 2 3; do
  for s in "64 2000 20" "128 5000 50"; do
    set -- $s
    echo "-- rep $rep, nprobe $1 / refine $2 / rerank $3: bytes+heap, then fastscan+reservoir"
    run "$I100" $1 $2 $3 results/phase2-100M-repeats.csv "deep100M $HOST v0.5 bytes+heap rep$rep" --nq 2000 --heap
    run "$F100" $1 $2 $3 results/phase2-100M-repeats.csv "deep100M $HOST v0.5 fastscan+reservoir rep$rep" --nq 2000
  done
done

stamp "D. 100M cold grid, new path"
echo "$HEAD"
for s in "64 2000 20" "64 5000 50" "128 2000 50" "128 5000 50" "256 5000 50" "256 10000 50"; do
  set -- $s
  run "$F100" $1 $2 $3 results/phase2-100M-grid.csv "deep100M $HOST v0.5 fastscan+reservoir cold" --nq 5000
done
$BIN ceiling --index "$F100" --queries "$Q" --gt "$G100" --nprobe 64,128,256 2>/dev/null

free -m
stamp "phase 2 done"
