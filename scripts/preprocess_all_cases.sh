#!/bin/bash
# Forwarding wrapper — the actual script lives in scripts/preprocessing/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/preprocessing/preprocess_all_cases.sh" "$@"
