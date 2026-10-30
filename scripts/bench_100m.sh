#!/usr/bin/env bash
# Deep 100M, step 2 of 2: benchmark from a COLD page cache.
#
# At 100M the full vectors (38 GB) don't fit in RAM, so reranking reads from the SSD. Whatever
# Linux has cached from a previous run makes the next one look faster. So every setting runs as
# its own process, after emptying the page cache, with no warm-up. That needs sudo: you'll be
# asked for your Kali password once.
#
# Per setting: latency is measured first (500 queries, one thread, cache cold), then recall and
# QPS (5,000 queries, all threads; the cache is partly warm by then, so QPS is mildly optimistic).
# Expect roughly 30-40 minutes.
set -euo pipefail
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

DATA=${DATA:-$HOME/binvec-data/deep}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$HOME/binvec-target}
export RUSTFLAGS=${RUSTFLAGS:-"-C target-cpu=native"}
BIN=$CARGO_TARGET_DIR/release/binvec
NLIST=${NLIST:-16384}
IDX="$DATA/idx-100M-$NLIST-rot-unbiased.bin"
LABEL="deep100M nlist=$NLIST $(uname -n) v0.3 cold"
CSV=results/results-100M-cold-v2.csv

cargo build --release
mkdir -p results

drop_caches() { sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }

sudo -v   # ask for the password now, not halfway through
# Keep sudo alive for the whole run (sudo forgets the password after ~15 minutes).
while true; do sudo -n true; sleep 60; done 2>/dev/null &
KEEPALIVE=$!
trap 'kill $KEEPALIVE 2>/dev/null' EXIT

for mode in "--prefetch" ""; do
  echo
  echo "== cold-cache benchmark ${mode:-without prefetch} =="
  echo "nprobe refine rerank recall@10   p50 ms   p99 ms  mean ms     QPS  |   cent   scan refine rerank"
  for np in 32 64 128; do
    for rr in 100 200 500 1000; do
      drop_caches
      $BIN bench --index "$IDX" --base "$DATA/base.100M.fbin" \
        --queries "$DATA/query.public.10K.fbin" --gt "$DATA/gt-100M.bin" \
        --nprobe $np --rerank $rr --nq-latency 500 --nq 5000 --warmup 0 $mode \
        --csv "$CSV" --label "$LABEL" 2>/dev/null | tail -1
    done
  done
done

echo
echo "== ceiling: share of true neighbours inside the probed clusters (no disk reads) =="
$BIN ceiling --index "$IDX" --queries "$DATA/query.public.10K.fbin" --gt "$DATA/gt-100M.bin" \
  --nprobe 16,32,64,128,256 | tee results/ceiling-100M.txt

echo
echo "columns: nprobe rerank recall@10 p50_ms p99_ms mean_ms QPS   (all rows also in $CSV)"
