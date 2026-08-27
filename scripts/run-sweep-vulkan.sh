#!/bin/bash
# Vulkan sweep for Qwen3.8-27B Q4_K_M @ 262K single R9700 — mirrors the ROCm sweep
cd ${WORKDIR:-.}

MODEL=${MODEL:-~/Models/qwen38-27b-q4km/Qwen3.8-27B-UD-Q4_K_M.gguf}
SRV=${SRV:-./build-vulkan/bin/llama-server}
HARNESS=${HARNESS:-./scripts/harness.py}
PORT=8091
export HIP_VISIBLE_DEVICES=99
export GGML_VK_VISIBLE_DEVICES=1

run() {
  LABEL=$1; shift
  FLAGS="$@"
  pkill -x llama-server 2>/dev/null || true; sleep 4
  $SRV -m $MODEL -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -np 1 \
    --port $PORT --no-webui $FLAGS \
    > server-$LABEL.log 2>&1 &
  SPID=$!
  echo "=== $LABEL (PID=$SPID) flags=$FLAGS" | tee row-$LABEL.out
  CTX=$(echo "$FLAGS" | grep -oP "(?<=-c )\d+" || echo 4096)
  timeout 540 python3 -u $HARNESS $PORT result-$LABEL.json $CTX 2>&1 | tee -a row-$LABEL.out
  RC=$?
  kill $SPID 2>/dev/null || true; wait $SPID 2>/dev/null || true
  pkill -x llama-server 2>/dev/null || true; sleep 3
  echo "=== $LABEL done rc=$RC" | tee -a row-$LABEL.out
}

# Same 8 rows as the ROCm sweep + the MTP3 -b1024/-ub1024 winner variant from the hipfire comparison
run 262k-nospec -c 262144
run 262k-mtp1 -c 262144 --spec-type draft-mtp --spec-draft-n-max 1
run 262k-mtp2 -c 262144 --spec-type draft-mtp --spec-draft-n-max 2
run 262k-mtp3 -c 262144 --spec-type draft-mtp --spec-draft-n-max 3
run 262k-mtp4 -c 262144 --spec-type draft-mtp --spec-draft-n-max 4
run 262k-mtp5 -c 262144 --spec-type draft-mtp --spec-draft-n-max 5
run 262k-ngram-simple -c 262144 --spec-type ngram-simple --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 4
run 262k-ngram-mod -c 262144 --spec-type ngram-mod --spec-ngram-mod-n-max 8 --spec-ngram-mod-n-match 24
run 262k-mtp3-ub1024 -c 262144 --spec-type draft-mtp --spec-draft-n-max 3 -b 1024 -ub 1024

# DFlash2 at 262K via z-lab fork Vulkan build (mirrors ROCm row result-262k-dflash4-fork)
LABEL=262k-dflash4
pkill -x llama-server 2>/dev/null || true; sleep 4
~/llama.cpp-dflash2/build-vulkan/bin/llama-server -m $MODEL -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -np 1 \
  --port $PORT --no-webui -c 262144 --spec-type draft-dflash \
  --model-draft ${DFLASH:-~/Models/qwen38-27b-dflash2/Qwen3.8-27B-DFlash2-Q8_0.gguf} --spec-draft-n-max 4 \
  > server-$LABEL.log 2>&1 &
SPID=$!
echo "=== $LABEL (PID=$SPID) dflash2-fork-vulkan" | tee row-$LABEL.out
timeout 540 python3 -u $HARNESS $PORT result-$LABEL.json 262144 2>&1 | tee -a row-$LABEL.out
RC=$?
kill $SPID 2>/dev/null || true; wait $SPID 2>/dev/null || true
pkill -x llama-server 2>/dev/null || true; sleep 3
echo "=== $LABEL done rc=$RC" | tee -a row-$LABEL.out

echo "ALL DONE VULKAN"
