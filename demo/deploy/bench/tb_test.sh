#!/bin/bash
# Prefill is compute-bound and scales to 8 threads (llama-bench: 108 -> 141 t/s),
# while generation plateaus at 4. --threads-batch splits the two. This is
# bit-identical math, only thread allocation -- no accuracy question at all.
UP=${NGEC_LLAMACPP_BIN:-$HOME/ngec-llamacpp/bin}
IK=${IK_LLAMACPP_BIN:-$HOME/ik_llama.cpp/build/bin}
M=${NGEC_GGUF:-$HOME/ngec-llamacpp/attr-exp5.1-q8.gguf}
LD_LIBRARY_PATH=$UP $UP/llama-server --model $M --host 127.0.0.1 --port 8081 \
  --ctx-size 8192 --threads 4 > "$(dirname "$0")"/tb_a.log 2>&1 & P1=$!
LD_LIBRARY_PATH=$UP $UP/llama-server --model $M --host 127.0.0.1 --port 8083 \
  --ctx-size 8192 --threads 4 --threads-batch 8 > "$(dirname "$0")"/tb_b.log 2>&1 & P2=$!
$IK/llama-server --model $M --host 127.0.0.1 --port 8082 \
  --ctx-size 8192 --threads 4 --threads-batch 8 > "$(dirname "$0")"/tb_c.log 2>&1 & P3=$!
trap 'kill $P1 $P2 $P3 2>/dev/null' EXIT
for p in 8081 8083 8082; do for i in $(seq 1 120); do curl -sf http://127.0.0.1:$p/health >/dev/null 2>&1 && break; sleep 1; done; done
echo "three servers up"
for r in 1 2 3; do
  python3 "$(dirname "$0")"/replay.py 8081 "up_t4      " 1 | grep rep0
  python3 "$(dirname "$0")"/replay.py 8083 "up_t4_tb8  " 1 | grep rep0
  python3 "$(dirname "$0")"/replay.py 8082 "ik_t4_tb8  " 1 | grep rep0
done
