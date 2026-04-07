#!/bin/bash
set -euo pipefail

# StateStrike Demo Launch Script
# Starts all services and runs 200-step agent demo

echo "🎯 StateStrike — OpenEnv Hackathon Demo"
echo "Starting honeypot API..."
uvicorn honeypot.app:app --port 8000 &
HONEY_PID=$!

echo "Starting OpenEnv environment server..."
python -m statestrike_env.server &
ENV_PID=$!

echo "Starting dashboard..."
streamlit run dashboard/app.py --server.port 8501 &
DASH_PID=$!

cleanup() {
  kill "$HONEY_PID" "$ENV_PID" "$DASH_PID" 2>/dev/null || true
}
trap cleanup EXIT

sleep 3
echo "All services up. Running agent..."
echo "Dashboard: http://localhost:8501"
python -m agent.runner --steps 200
