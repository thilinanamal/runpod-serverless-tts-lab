#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON="$PROJECT_DIR/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
  python3 -m venv "$PROJECT_DIR/.venv"
fi

"$PYTHON" -m pip install -q -r "$PROJECT_DIR/ui/requirements.txt"
exec "$PYTHON" "$PROJECT_DIR/ui/app.py"
