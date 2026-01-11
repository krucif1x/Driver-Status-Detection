#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/home/cc/Downloads/dd_2025"
VENV_DIR="$APP_DIR/venv"
SERVICE_NAME="driver-status-detection.service"

MODE="${1:-manual}"

if [[ "$MODE" == "service" ]]; then
  cd "$APP_DIR"
  source "$VENV_DIR/bin/activate"
  exec python3 main.py
fi

echo "Starting driver status detection..."

if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files | grep -q "^${SERVICE_NAME}"; then
  sudo systemctl start "$SERVICE_NAME"
  sudo systemctl status "$SERVICE_NAME" --no-pager -n 10 || true
  exit 0
fi

cd "$APP_DIR"
source "$VENV_DIR/bin/activate"
exec python3 main.py