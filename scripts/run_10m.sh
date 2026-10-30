#!/usr/bin/env bash
# Build and benchmark on the Deep 10M subset.
# Run from the project folder after scripts/download_deep.sh 10 has finished.
set -euo pipefail
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"   # make sure cargo is on PATH

DATA=${DATA:-$HOME/binvec-data/deep}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$HOME/binvec-target}   # keep build output on Linux disk
export RUSTFLAGS=${RUSTFLAGS:-"-C target-cpu=native"}               # use this CPU's AVX-512
BIN=$CARGO_TARGET_DIR/release/binvec
NLIST=${NLIST:-4096}

cargo build --release
mkdir -p results

echo "== 1/3 sanity check: brute force vs published ground truth =="
$BIN verify --base "$DATA/base.10M.fbin" --queries "$DATA/query.public.10K.fbin" --gt "$DATA/gt-10M.bin" --nq 50

echo "== 2/3 build index (nlist=$NLIST) =="
[ -s "$DATA/idx-10M-$NLIST-rot-unbiased.bin" ] || \
  $BIN build --base "$DATA/base.10M.fbin" --nlist "$NLIST" --train-n 500000 --out "$DATA/idx-10M-$NLIST-rot-unbiased.bin"

echo "== 3/3 benchmark =="
$BIN bench --index "$DATA/idx-10M-$NLIST-rot-unbiased.bin" --base "$DATA/base.10M.fbin" \
  --queries "$DATA/query.public.10K.fbin" --gt "$DATA/gt-10M.bin" \
  --nprobe 8,16,32,64,128 --rerank 0,100,200,500 \
  --csv results/results-10M-v2.csv --label "deep10M nlist=$NLIST $(uname -n) v0.2"
