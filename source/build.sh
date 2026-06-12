#!/bin/sh
set -e

PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/opt/python@3.9/bin/python3.9}"

cd "$(dirname "$0")/.."
"$PYTHON_BIN" source/build_variable.py --config source/build-variable.config.json
