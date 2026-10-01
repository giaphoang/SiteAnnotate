#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if git rev-parse --show-toplevel >/dev/null 2>&1; then
  git submodule update --init --recursive
fi

mkdir -p vendor

while IFS='|' read -r name url ref; do
  [ -n "$name" ] || continue
  path="vendor/$name"
  if [ -d "$path/.git" ] || [ -f "$path/.git" ]; then
    echo "Using existing $name checkout at $path"
    continue
  fi
  if [ -e "$path" ] && [ "$(find "$path" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    echo "Refusing to overwrite non-empty source directory: $path" >&2
    exit 1
  fi
  echo "Cloning $name ($ref)"
  git clone --branch "$ref" --depth 1 "$url" "$path"
done <<'SOURCES'
paidf-curation-and-retrieval|https://github.com/NVIDIA/paidf-curation-and-retrieval.git|main
cosmos-reason1|https://github.com/nvidia-cosmos/cosmos-reason1.git|main
SUSTechPOINTS|https://github.com/naurril/SUSTechPOINTS.git|dev-auto-annotate
RELLIS-3D|https://github.com/unmannedlab/RELLIS-3D.git|main
SOURCES

python3 - <<'PY'
import json
import subprocess
from pathlib import Path

lock_path = Path("sources.lock.json")
data = json.loads(lock_path.read_text())
for name, source in data["sources"].items():
    path = Path("vendor") / name
    if not (path / ".git").exists():
        continue
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    source["commit"] = commit
lock_path.write_text(json.dumps(data, indent=2) + "\n")
PY

echo "Approved sources initialized and sources.lock.json updated."
