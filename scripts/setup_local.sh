#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo "Local setup complete."
echo "All dependencies are installed in .venv."
echo "Model caches will be stored under ./models at runtime."
