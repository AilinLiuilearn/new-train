#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [[ -n "${MYENV_PYTHON:-}" ]]; then
  exec "$MYENV_PYTHON" -u "$@"
fi
if command -v conda >/dev/null 2>&1; then
  exec conda run --no-capture-output -n myenv python -u "$@"
fi
if [[ -x /root/miniconda3/envs/myenv/bin/python ]]; then
  exec /root/miniconda3/envs/myenv/bin/python -u "$@"
fi
echo 'myenv not found; create it using environment.yml or set MYENV_PYTHON' >&2
exit 1
