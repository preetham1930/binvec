#!/usr/bin/env bash
# Deep 100M, step 1 of 2: check the data and build the index. No password needed, so this can
# run unattended (e.g. overnight). Expect roughly 30-45 minutes on an 8-core laptop.
#
# Run after:  scripts/download_deep.sh 100
# Then run:   scripts/bench_100m.sh
set -euo pipefail
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

DATA=${DATA:-$HOME/binvec-data/deep}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$HOME/binvec-target}
export RUSTFLAGS=${RUSTFLAGS:-"-C target-cpu=native"}
BIN=$CARGO_TARGET_DIR/release/binvec
NLIST=${NLIST:-16384}   # ~sqrt(100M) rounded up to a power of two: ~6,100 vectors per list

EXPECTED=$((8 + 100000000 * 96 * 4))
ACTUAL=$(stat -c%s "$DATA/base.100M.fbin")
if [ "$ACTUAL" -ne "$EXPECTED" ]; then
  echo "base.100M.fbin is $ACTUAL bytes, expected $EXPECTED. Re-run scripts/download_deep.sh 100"
  exit 1
fi

cargo build --release
mkdir -p results

echo "== 1/2 sanity check: brute force vs published ground truth (one pass over 38 GB) =="
$BIN verify --base "$DATA/base.100M.fbin" --queries "$DATA/query.public.10K.fbin" --gt "$DATA/gt-100M.bin" --nq 50

echo "== 2/2 build index (nlist=$NLIST) =="
OUT="$DATA/idx-100M-$NLIST-rot-unbiased.bin"
if [ -s "$OUT" ]; then
  echo "index already exists: $OUT"
else
  TIMER=""
  [ -x /usr/bin/time ] && TIMER="/usr/bin/time -v"   # also reports peak memory, if installed
  $TIMER $BIN build --base "$DATA/base.100M.fbin" --nlist "$NLIST" --train-n 1000000 --out "$OUT" \
    2>&1 | tee results/build-100M.log
  grep -E "Maximum resident|Elapsed" results/build-100M.log || true
fi
echo "== build done. Next: bash scripts/bench_100m.sh =="
