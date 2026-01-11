#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="driver-status-detection.service"

echo "Stopping driver status detection..."

# If systemd service exists, use it
if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files | grep -q "^${SERVICE_NAME}"; then
  sudo systemctl stop "$SERVICE_NAME"
  sudo systemctl status "$SERVICE_NAME" --no-pager -n 10 || true
  exit 0
fi

# Fallback: stop manual run
PID="$(pgrep -f "python(3)? .*main\.py" || true)"
if [[ -n "${PID}" ]]; then
  echo "Stopping process PID(s): ${PID}"
  kill ${PID} || true
  sleep 1
  PID2="$(pgrep -f "python(3)? .*main\.py" || true)"
  if [[ -n "${PID2}" ]]; then
    echo "Force killing PID(s): ${PID2}"
    kill -9 ${PID2} || true
  fi
  echo "Stopped."
else
  echo "No driver status detection process found."
fi