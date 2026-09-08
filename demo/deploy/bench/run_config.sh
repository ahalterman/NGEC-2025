#!/bin/bash
# Start a test llama-server on :8081 with the given flags, replay the real
# extraction prompts against it, then shut it down.
#   run_config.sh LABEL [extra llama-server flags...]
set -u
LABEL="$1"; shift
BIN=~/ngec-llamacpp/bin
export LD_LIBRARY_PATH=$BIN

$BIN/llama-server --model ~/ngec-llamacpp/attr-exp5.1-q8.gguf \
    --host 127.0.0.1 --port 8081 "$@" > ~/bench/server_$LABEL.log 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT

for i in $(seq 1 120); do
    curl -sf http://127.0.0.1:8081/health >/dev/null 2>&1 && break
    kill -0 $PID 2>/dev/null || { echo "$LABEL: server died"; tail -5 ~/bench/server_$LABEL.log; exit 1; }
    sleep 1
done

python3 ~/bench/replay.py 8081 "$LABEL" "${REPS:-2}"
