#!/bin/bash
set -euo pipefail

# StateStrike bootstrap script
# Creates local env file and installs pinned dependencies.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Ensure Docker bind-mount file targets exist as files.
touch statestrike.db
touch telemetry.json

echo "StateStrike environment setup complete."
