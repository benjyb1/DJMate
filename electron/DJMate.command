#!/bin/bash
# Double-click launcher for DJMate on macOS.
# Right-click → Open the first time (unsigned scripts need one Gatekeeper OK).
cd "$(dirname "$0")" || exit 1

if [ ! -d node_modules ]; then
  echo "First run — installing Electron (one-off, ~1 min)…"
  npm install || { echo "npm install failed"; read -r -p "Press return to close"; exit 1; }
fi

exec npm start
