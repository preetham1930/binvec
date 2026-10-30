#!/usr/bin/env bash
# v0.2 experiment on Deep 10M: does a random rotation and/or the unbiased (RaBitQ) estimator
# fix the weak 1-bit ranking seen in v0.1?
#
# Builds four indexes (rotation on/off x estimator ls/unbiased) and benchmarks each, then
# measures the "ceiling": the recall you would get if every vector in the probed clusters were
# scored exactly. That separates two causes of lost recall:
#   - clustering (true neighbours sit in clusters we never probe)  -> ceiling is low
#   - the 1-bit estimate (they are probed but ranked too low)      -> ceiling is high
#
# Takes roughly 15-20 minutes on an 8-core laptop.
set -euo pipefail
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

DATA=${DATA:-$HOME/binvec-data/deep}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$HOME/binvec-target}
export RUSTFLAGS=${RUSTFLAGS:-"-C target-cpu=native"}
BIN=$CARGO_TARGET_DIR/release/binvec
NLIST=4096
COMMON=(--base "$DATA/base.10M.fbin" --queries "$DATA/query.public.10K.fbin" --gt "$DATA/gt-10M.bin")
LABEL="deep10M nlist=$NLIST $(uname -n) v0.2"

cargo build --release
mkdir -p results
rm -f "$DATA/idx-10M-$NLIST.bin"   # v0.1 index format, no longer readable

build() {  # $1 = file suffix, rest = build flags
  local out="$DATA/idx-10M-$NLIST-$1.bin"; shift
  [ -s "$out" ] || $BIN build --base "$DATA/base.10M.fbin" --nlist $NLIST --train-n 500000 "$@" --out "$out"
}

echo "== building 4 variants =="
build norot-ls       --no-rotate --estimator ls
build norot-unbiased --no-rotate --estimator unbiased
build rot-ls         --estimator ls
build rot-unbiased   --estimator unbiased

for v in norot-ls norot-unbiased rot-ls rot-unbiased; do
  echo
  echo "== benchmark: $v =="
  $BIN bench --index "$DATA/idx-10M-$NLIST-$v.bin" "${COMMON[@]}" \
    --nprobe 16,32,64 --rerank 0,100,200,500 --nq-latency 1000 \
    --csv results/ablation.csv --label "$LABEL"
done

echo
echo "== ceiling: every probed vector scored exactly (slow on purpose, fewer queries) =="
$BIN bench --index "$DATA/idx-10M-$NLIST-rot-unbiased.bin" "${COMMON[@]}" \
  --nprobe 16,32,64,128 --rerank 1000000 --nq-latency 100 \
  --csv results/ceiling.csv --label "$LABEL ceiling"
