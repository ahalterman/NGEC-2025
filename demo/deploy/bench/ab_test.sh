#!/bin/bash
# Alternate upstream and ik on the same prompts so slow drift in the box hits
# both equally. Both servers stay loaded; only one is serving at any moment.
UP=${NGEC_LLAMACPP_BIN:-$HOME/ngec-llamacpp/bin}
IK=${IK_LLAMACPP_BIN:-$HOME/ik_llama.cpp/build/bin}
M=${NGEC_GGUF:-$HOME/ngec-llamacpp/attr-exp5.1-q8.gguf}

LD_LIBRARY_PATH=$UP $UP/llama-server --model $M --host 127.0.0.1 --port 8081 \
  --ctx-size 8192 --threads 4 > "$(dirname "$0")"/ab_up.log 2>&1 &
P1=$!
$IK/llama-server --model $M --host 127.0.0.1 --port 8082 \
  --ctx-size 8192 --threads 4 > "$(dirname "$0")"/ab_ik.log 2>&1 &
P2=$!
trap 'kill $P1 $P2 2>/dev/null' EXIT

for p in 8081 8082; do
  for i in $(seq 1 120); do
    curl -sf http://127.0.0.1:$p/health >/dev/null 2>&1 && break
    sleep 1
  done
done
echo "both servers up"
for r in 1 2 3 4; do
  python3 "$(dirname "$0")"/replay.py 8081 upstream 1 | grep rep0
  python3 "$(dirname "$0")"/replay.py 8082 ik       1 | grep rep0
done
