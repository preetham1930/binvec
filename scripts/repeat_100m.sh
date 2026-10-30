#!/usr/bin/env bash
# Deep 100M: repeat the headline settings 3 times each, cold cache, with prefetch, and report
# the median. Single runs on a laptop under WSL2 vary a lot (Windows background work, heat,
# SSD state), so a published number should be a median, with the spread shown.
# Asks for your Kali password once. Expect roughly 15-20 minutes.
set -euo pipefail
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

DATA=${DATA:-$HOME/binvec-data/deep}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$HOME/binvec-target}
export RUSTFLAGS=${RUSTFLAGS:-"-C target-cpu=native"}
BIN=$CARGO_TARGET_DIR/release/binvec
NLIST=${NLIST:-16384}
IDX="$DATA/idx-100M-$NLIST-rot-unbiased.bin"
REPS=${REPS:-3}
CSV=results/repeats-100M.csv
LABEL="deep100M nlist=$NLIST $(uname -n) v0.3 cold repeat"

cargo build --release
mkdir -p results
drop_caches() { sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
sudo -v
while true; do sudo -n true; sleep 60; done 2>/dev/null &
KEEPALIVE=$!
trap 'kill $KEEPALIVE 2>/dev/null' EXIT

SUMMARY=""
for cfg in "32 500" "64 500" "128 500" "128 1000"; do
  set -- $cfg
  np=$1; rr=$2
  echo
  echo "== nprobe $np, rerank $rr: $REPS cold runs =="
  echo " nprobe  rerank recall@10   p50 ms   p99 ms  mean ms        QPS"
  ROWS=""
  for rep in $(seq 1 "$REPS"); do
    drop_caches
    ROW=$($BIN bench --index "$IDX" --base "$DATA/base.100M.fbin" \
      --queries "$DATA/query.public.10K.fbin" --gt "$DATA/gt-100M.bin" \
      --nprobe "$np" --rerank "$rr" --nq-latency 500 --nq 2000 --warmup 0 --prefetch \
      --csv "$CSV" --label "$LABEL" 2>/dev/null | tail -1)
    echo "$ROW"
    ROWS+="$ROW"$'\n'
  done
  # Median and range of p50 (column 4) and p99 (column 5) across the runs.
  # (sort + plain awk, so it works with any awk, not only GNU awk.)
  REC=$(printf '%s' "$ROWS" | awk 'NF {r=$3} END {print r}')
  MID=$(( (REPS + 1) / 2 ))
  P50=$(printf '%s' "$ROWS" | awk 'NF {print $4}' | sort -g | sed -n "${MID}p")
  P99S=$(printf '%s' "$ROWS" | awk 'NF {print $5}' | sort -g)
  P99=$(printf '%s\n' "$P99S" | sed -n "${MID}p")
  LO=$(printf '%s\n' "$P99S" | head -1)
  HI=$(printf '%s\n' "$P99S" | tail -1)
  SUMMARY+=$(printf "%7s %7s %9s %9.1f %9.1f %9.1f-%-9.1f" "$np" "$rr" "$REC" "$P50" "$P99" "$LO" "$HI")$'\n'
done

echo
echo "== summary: median of $REPS cold runs (prefetch on) =="
printf "%7s %7s %9s %9s %9s %19s\n" nprobe rerank recall@10 "p50 ms" "p99 ms" "p99 range ms"
printf '%s' "$SUMMARY" | tee results/repeats-100M-summary.txt
