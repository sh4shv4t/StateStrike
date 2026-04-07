#!/bin/bash
set -e

echo "[StateStrike] Starting honeypot on port 8000..."
uvicorn honeypot.app:app --host 0.0.0.0 --port 8000 &
HONEYPOT_PID=$!

echo "[StateStrike] Waiting for honeypot..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "[StateStrike] Honeypot ready."
    break
  fi
  sleep 1
done

echo "[StateStrike] Starting environment server on port 7860..."
export HONEYPOT_URL="http://localhost:8000"
uvicorn statestrike_env.environment:app --host 0.0.0.0 --port 7860

wait $HONEYPOT_PID
