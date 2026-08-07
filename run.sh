#!/usr/bin/env bash

set -euo pipefail

echo "Use run_tum_dynamic.sh for portable TUM RGB-D execution."
exec "$(dirname "$0")/run_tum_dynamic.sh" "$@"
