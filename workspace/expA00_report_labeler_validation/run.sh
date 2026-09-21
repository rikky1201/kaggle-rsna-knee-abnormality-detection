#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python build_validation_set.py "$@"
