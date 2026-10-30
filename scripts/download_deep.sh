#!/usr/bin/env bash
# Download the Deep1B subset used by the NeurIPS'21 Big-ANN benchmark.
#
#   scripts/download_deep.sh 10    # first 10M vectors  (~3.8 GB)
#   scripts/download_deep.sh 100   # first 100M vectors (~38 GB)
#
# Only the first N vectors of the 1B file are fetched, using HTTP range requests, in 512 MB
# chunks. If the download is interrupted, run the same command again: it resumes from the
# last complete chunk.
set -euo pipefail

N_M=${1:-10}
DIR=${2:-$HOME/binvec-data/deep}
D=96
BASE_URL=${BASE_URL:-https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP}
GT_URL=${GT_URL:-https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks}
CHUNK=${CHUNK:-$((512 * 1024 * 1024))}

case "$N_M" in
  10|100) ;;
  *) echo "Published ground truth exists only for 10 and 100 (million). Got: $N_M"; exit 1 ;;
esac

mkdir -p "$DIR"
cd "$DIR"
N=${TEST_N:-$((N_M * 1000000))}   # TEST_N only for testing the script

echo "== queries (10K) =="
[ -s query.public.10K.fbin ] || curl -fL --retry 5 -o query.public.10K.fbin "$BASE_URL/query.public.10K.fbin"

echo "== ground truth for ${N_M}M =="
[ -s "gt-${N_M}M.bin" ] || curl -fL --retry 5 -o "gt-${N_M}M.bin" "$GT_URL/GT_${N_M}M/deep-${N_M}M"

OUT="base.${N_M}M.fbin"
DATA_BYTES=$((N * D * 4))
echo "== base vectors: first ${N_M}M ($((DATA_BYTES / 1000000)) MB) =="

if [ ! -e "$OUT" ] && [ "$N_M" = 100 ] && [ -s base.10M.fbin ] && [ "$(stat -c%s base.10M.fbin)" -eq $((8 + 10000000 * D * 4)) ]; then
  # The 10M file is exactly the first 10% of the 100M file: reuse it instead of downloading
  # those 3.8 GB again. Copy, then rewrite the 8-byte header to say 100M.
  echo "  reusing base.10M.fbin as the first 10%..."
  cp base.10M.fbin "$OUT.tmp"
  perl -e "print pack('VV', $N, $D)" | dd of="$OUT.tmp" bs=8 count=1 conv=notrunc status=none
  mv "$OUT.tmp" "$OUT"
fi

if [ ! -e "$OUT" ]; then
  # 8-byte header: n and d as little-endian u32.
  perl -e "print pack('VV', $N, $D)" > "$OUT"
fi

while true; do
  HAVE=$(( $(stat -c%s "$OUT") - 8 ))
  if [ "$HAVE" -ge "$DATA_BYTES" ]; then break; fi
  LEFT=$((DATA_BYTES - HAVE))
  LEN=$(( LEFT < CHUNK ? LEFT : CHUNK ))
  START=$((8 + HAVE))
  END=$((START + LEN - 1))
  echo "  $((HAVE * 100 / DATA_BYTES))% done, fetching next $((LEN / 1048576)) MB..."
  CODE=$(curl -sS -L --retry 5 --retry-delay 5 -r "$START-$END" -o chunk.part -w '%{http_code}' "$BASE_URL/base.1B.fbin") || CODE="$CODE, connection dropped"
  if [ "$CODE" = "206" ] && [ -f chunk.part ] && [ "$(stat -c%s chunk.part)" -eq "$LEN" ]; then
    cat chunk.part >> "$OUT"
    rm -f chunk.part
  else
    echo "  chunk failed (HTTP $CODE), retrying in 10s..."
    rm -f chunk.part
    sleep 10
  fi
done

echo "== done =="
ls -lh "$DIR"
