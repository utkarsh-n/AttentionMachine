#!/bin/zsh
set -euo pipefail

task_root="${0:A:h:h}"
task_cache_root="${TMPDIR:-/tmp}/attention-machine-core"
mkdir -p "$task_cache_root"
cd "$task_root/AttentionCore"

export CLANG_MODULE_CACHE_PATH="$task_cache_root/clang"
export SWIFTPM_MODULECACHE_OVERRIDE="$task_cache_root/swift"
swift test --disable-sandbox \
  --scratch-path "$task_cache_root/build" \
  --cache-path "$task_cache_root/package-cache" \
  --config-path "$task_cache_root/package-config" \
  --security-path "$task_cache_root/package-security"
