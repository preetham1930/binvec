#!/usr/bin/env bash
# Phase 4: binvec vs FAISS, Deep 10M, same machine, same queries, same ground truth, same
# measurement (latency: one query per call on one thread; QPS: all cores). Data in RAM for both.
# No password needed. Roughly 30-45 minutes. First run installs FAISS into ~/binvec-venv.
set -euo pipefail
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

DATA=${DATA:-$HOME/binvec-data/deep}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$HOME/binvec-target}
export RUSTFLAGS=${RUSTFLAGS:-"-C target-cpu=native"}
BIN=$CARGO_TARGET_DIR/release/binvec
VENV=${VENV:-$HOME/binvec-venv}
HOST=$(uname -n)
Q="$DATA/query.public.10K.fbin"; G="$DATA/gt-10M.bin"; B10="$DATA/base.10M.fbin"
F10="$DATA/idx-10M-4096-fastscan.bin"; R10="$DATA/refine-10M-4096.bin"
stamp() { echo; echo "######## $* ($(date +%H:%M))"; }

stamp "setup: FAISS in a virtual environment"
# A half-made environment (python but no pip, left by a failed first attempt) is rebuilt.
if [ ! -x "$VENV/bin/pip" ]; then
  rm -rf "$VENV"
  python3 -m venv "$VENV" || { echo "python3 -m venv failed: run 'sudo apt install -y python3-venv' and retry"; exit 1; }
fi
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q "faiss-cpu==1.15.1" numpy
"$VENV/bin/python" -c "import faiss, numpy; print('faiss', faiss.__version__, 'numpy', numpy.__version__)"
cargo build --release
mkdir -p results
nproc; free -m

cat "$B10" "$R10" "$F10" > /dev/null

stamp "binvec, Deep 10M"
$BIN bench --index "$F10" --base "$B10" --queries "$Q" --gt "$G" --refine "$R10" \
  --nprobe 32,64,128,256 --refine-k 1000,2000,5000 --rerank 20,50 --nq-latency 1000 \
  --csv results/phase4-binvec-10M.csv --label "deep10M $HOST v0.6" 2>/dev/null

stamp "FAISS, Deep 10M (builds 5 indexes; each prints its rows as it goes)"
"$VENV/bin/python" -I scripts/faiss_compare.py --base "$B10" --queries "$Q" --gt "$G" \
  --nlist 4096 --nprobe 32,64,128,256 --k-factor 10,20,50,100,200 --nq-latency 1000 \
  --tmp-dir "$DATA" --csv results/phase4-faiss-10M.csv --label "deep10M $HOST"

free -m
stamp "phase 4 done"
