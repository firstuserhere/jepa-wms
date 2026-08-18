#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

: "${GIT_COMMIT_HASH:?SkyPilot must clone an exact --git-ref commit}"
if [[ ! "$GIT_COMMIT_HASH" =~ ^[0-9a-f]{40}$ ]] || [[ ! -d .git ]]; then
  echo "Smoke source must be a SkyPilot Git workdir pinned to a full commit" >&2
  exit 1
fi
test "$(git rev-parse HEAD)" = "$GIT_COMMIT_HASH"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

uv venv --python 3.10
uv pip install --python .venv/bin/python "torch==2.7.0" wandb
