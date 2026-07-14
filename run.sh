#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIXI_EXE="${PIXI_PATH:-}"

if [[ -z "$PIXI_EXE" && -x "$PROJECT_DIR/../bin/pixi" ]]; then
  PIXI_EXE="$PROJECT_DIR/../bin/pixi"
fi
if [[ -z "$PIXI_EXE" && -x "$HOME/.pixi/bin/pixi" ]]; then
  PIXI_EXE="$HOME/.pixi/bin/pixi"
fi
if [[ -z "$PIXI_EXE" ]] && command -v pixi >/dev/null 2>&1; then
  PIXI_EXE="$(command -v pixi)"
fi
if [[ -z "$PIXI_EXE" ]]; then
  echo "Pixi was not found. Install it from https://pixi.sh or set PIXI_PATH." >&2
  exit 1
fi

cd "$PROJECT_DIR"
exec python3 run.py --pixi-path "$PIXI_EXE" "$@"
