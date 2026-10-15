#!/bin/zsh
set -euo pipefail
runtime_dir="${0:A:h}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/private/tmp/attention-uv-cache}"
export HF_HOME="$runtime_dir/.hf-cache"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_XET=1
if [[ ! -x "$runtime_dir/.venv/bin/python" ]]; then
  uv venv --python 3.12 "$runtime_dir/.venv"
fi
uv pip install --python "$runtime_dir/.venv/bin/python" -r "$runtime_dir/requirements.txt"
"$runtime_dir/.venv/bin/hf" download mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --revision 50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b \
  --local-dir "$runtime_dir/models/Qwen3-4B-Instruct-2507-4bit" \
  --exclude .gitattributes --max-workers 3
"$runtime_dir/.venv/bin/python" "$runtime_dir/capture_trace.py" \
  --prompt "Explain transformer attention in one short sentence."
