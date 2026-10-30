#!/usr/bin/env bash
# Phase 3: filtered search. One password prompt, then unattended. Roughly 30-45 minutes.
#
# Labels with planted selectivities (50%, 10%, 1%, 0.1%, 0.01%), two kinds:
#   uniform    - independent of the vectors (easy for clustering)
#   correlated - each label is a slab of space (a query's filtered neighbours are often far away)
# Strategies: post (search, then filter: the baseline), scan (filter inside the scan and keep
# probing), postings (inverted index: score only the matching vectors), auto (planner picks).
#
#  A. 10M: labels, exact filtered ground truth, all strategies, in-scan budget sweep
#  B. 100M: labels, exact filtered ground truth (one pass over 38 GB each)
#  C. 100M cold: every strategy in its own cold process
set -euo pipefail
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

DATA=${DATA:-$HOME/binvec-data/deep}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$HOME/binvec-target}
export RUSTFLAGS=${RUSTFLAGS:-"-C target-cpu=native"}
BIN=$CARGO_TARGET_DIR/release/binvec
HOST=$(uname -n)
Q="$DATA/query.public.10K.fbin"

F10="$DATA/idx-10M-4096-fastscan.bin";   R10="$DATA/refine-10M-4096.bin";   B10="$DATA/base.10M.fbin"
F100="$DATA/idx-100M-16384-fastscan.bin"; R100="$DATA/refine-100M-16384.bin"; B100="$DATA/base.100M.fbin"

cargo build --release
cargo test --release -q 2>&1 | grep "test result" | head -1
mkdir -p results
drop_caches() { sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
sudo -v
while true; do sudo -n true; sleep 60; done 2>/dev/null &
KEEPALIVE=$!
trap 'kill $KEEPALIVE 2>/dev/null' EXIT
stamp() { echo; echo "######## $* ($(date +%H:%M))"; }
free -m

# ---------------------------------------------------------------- 10M
stamp "A1. 10M labels and filtered ground truth"
for kind in uniform correlated; do
  L="$DATA/labels-10M-$kind.bin"; G="$DATA/gt-10M-filtered-$kind.bin"
  [ -s "$L" ] || $BIN labels --base "$B10" --kind $kind --out "$L"
  [ -s "$G" ] || $BIN filter-gt --base "$B10" --queries "$Q" --labels "$L" --nq 1000 --out "$G"
done

stamp "A2. 10M, all strategies (nprobe 128 / refine 5000 / rerank 50 / budget 20000)"
cat "$B10" "$R10" "$F10" > /dev/null
for kind in uniform correlated; do
  echo "-- $kind"
  $BIN fbench --index "$F10" --base "$B10" --refine "$R10" --queries "$Q" \
    --labels "$DATA/labels-10M-$kind.bin" --gt "$DATA/gt-10M-filtered-$kind.bin" \
    --mode post,scan,postings,auto --nprobe 128 --refine-k 5000 --rerank 50 --budget 20000 \
    --prefetch --csv results/phase3-10M.csv --label "deep10M $HOST v0.6 $kind" 2>/dev/null
done

stamp "A3. 10M, in-scan budget and planner constant"
for kind in uniform correlated; do
  echo "-- $kind: scan, budget sweep"
  $BIN fbench --index "$F10" --base "$B10" --refine "$R10" --queries "$Q" \
    --labels "$DATA/labels-10M-$kind.bin" --gt "$DATA/gt-10M-filtered-$kind.bin" \
    --mode scan --nprobe 128 --refine-k 5000 --rerank 50 --budget 5000,50000 \
    --prefetch --csv results/phase3-10M-sweep.csv --label "deep10M $HOST v0.6 $kind" 2>/dev/null
  for pc in 3 30; do
    echo "-- $kind: auto, postings cost $pc"
    $BIN fbench --index "$F10" --base "$B10" --refine "$R10" --queries "$Q" \
      --labels "$DATA/labels-10M-$kind.bin" --gt "$DATA/gt-10M-filtered-$kind.bin" \
      --mode auto --nprobe 128 --refine-k 5000 --rerank 50 --budget 20000 --postings-cost $pc \
      --prefetch --csv results/phase3-10M-sweep.csv --label "deep10M $HOST v0.6 $kind pc$pc" 2>/dev/null
  done
done

# ---------------------------------------------------------------- 100M
stamp "B. 100M labels and filtered ground truth"
for kind in uniform correlated; do
  L="$DATA/labels-100M-$kind.bin"; G="$DATA/gt-100M-filtered-$kind.bin"
  [ -s "$L" ] || $BIN labels --base "$B100" --kind $kind --out "$L"
  [ -s "$G" ] || $BIN filter-gt --base "$B100" --queries "$Q" --labels "$L" --nq 500 --out "$G"
done

stamp "C. 100M cold, one strategy per cold process (nprobe 256 / refine 5000 / rerank 50)"
for kind in uniform correlated; do
  for m in post scan postings auto; do
    echo "-- $kind, $m"
    drop_caches
    $BIN fbench --index "$F100" --base "$B100" --refine "$R100" --queries "$Q" \
      --labels "$DATA/labels-100M-$kind.bin" --gt "$DATA/gt-100M-filtered-$kind.bin" \
      --mode $m --nprobe 256 --refine-k 5000 --rerank 50 --budget 20000 --nq-latency 100 \
      --skip-postings-above 20000000 --prefetch \
      --csv results/phase3-100M-cold.csv --label "deep100M $HOST v0.6 $kind cold" 2>/dev/null
  done
done

free -m
stamp "phase 3 done"
