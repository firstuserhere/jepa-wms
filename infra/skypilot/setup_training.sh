#!/usr/bin/env bash
set -euo pipefail

readonly DINOV3_REPO_REVISION="54694f7627fd815f62a5dcc82944ffa6153bbb76"
readonly DINOV3_WEIGHTS_FILE="dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

: "${DINOV3_WEIGHTS_SHA256:?Set DINOV3_WEIGHTS_SHA256 to the full artifact SHA-256 digest}"

if [[ -n "${DINOV3_WEIGHTS_SOURCE_PATH:-}" ]]; then
  if [[ "$DINOV3_WEIGHTS_SOURCE_PATH" == *[$' \t\r\n']* ]] || [[ ! -f "$DINOV3_WEIGHTS_SOURCE_PATH" ]]; then
    echo "DINOV3_WEIGHTS_SOURCE_PATH must name a readable local/PVC artifact without whitespace" >&2
    exit 1
  fi
elif [[ ! "${DINOV3_WEIGHTS_URI:-}" =~ ^gs://[^[:space:]]+$ ]]; then
  echo "Provide either DINOV3_WEIGHTS_SOURCE_PATH or a non-secret gs:// DINOV3_WEIGHTS_URI" >&2
  exit 1
fi
if [[ ! "$DINOV3_WEIGHTS_SHA256" =~ ^8aa4cbdd[[:xdigit:]]{56}$ ]]; then
  echo "DINOV3_WEIGHTS_SHA256 must be the full 64-hex digest matching the official 8aa4cbdd filename identity" >&2
  exit 1
fi
DINOV3_WEIGHTS_SHA256="$(printf '%s' "$DINOV3_WEIGHTS_SHA256" | tr '[:upper:]' '[:lower:]')"
export DINOV3_WEIGHTS_SHA256

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

: "${GIT_COMMIT_HASH:?SkyPilot must clone an exact --git-ref commit}"
if [[ ! "$GIT_COMMIT_HASH" =~ ^[0-9a-f]{40}$ ]] || [[ ! -d .git ]]; then
  echo "Training source must be a SkyPilot Git workdir pinned to a full commit" >&2
  exit 1
fi
test "$(git rev-parse HEAD)" = "$GIT_COMMIT_HASH"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

# DROID does not import the legacy simulator-only Metaworld/D4RL packages.
# Everything else is installed exactly from the committed universal lock.
uv sync --frozen --extra dev \
  --no-install-package metaworld \
  --no-install-package d4rl

if [[ ! -d dinov3/.git ]]; then
  git clone --filter=blob:none https://github.com/facebookresearch/dinov3.git dinov3
fi
git -C dinov3 fetch --depth=1 origin "$DINOV3_REPO_REVISION"
git -C dinov3 checkout --detach "$DINOV3_REPO_REVISION"
test "$(git -C dinov3 rev-parse HEAD)" = "$DINOV3_REPO_REVISION"

mkdir -p .artifacts/dinov3
weights_path=".artifacts/dinov3/$DINOV3_WEIGHTS_FILE"
weights_sha256=""
if [[ -f "$weights_path" ]]; then
  weights_sha256="$(sha256sum "$weights_path" | awk '{print $1}')"
fi
if [[ "$weights_sha256" != "$DINOV3_WEIGHTS_SHA256" ]]; then
  temporary_weights="$(mktemp ".artifacts/dinov3/.${DINOV3_WEIGHTS_FILE}.XXXXXX")"
  trap 'rm -f "$temporary_weights"' EXIT
  # The current public Hub conversion exposes model.safetensors, not the
  # original Meta .pth key layout consumed by JEPA-WM.  Require a caller-owned
  # copy of the exact artifact and never silently substitute formats.
  if [[ -n "${DINOV3_WEIGHTS_SOURCE_PATH:-}" ]]; then
    cp "$DINOV3_WEIGHTS_SOURCE_PATH" "$temporary_weights"
  else
    if ! command -v gsutil >/dev/null 2>&1; then
      uv tool install --force gsutil
    fi
    gsutil -q cp "$DINOV3_WEIGHTS_URI" "$temporary_weights"
  fi
  weights_sha256="$(sha256sum "$temporary_weights" | awk '{print $1}')"
  if [[ "$weights_sha256" != "$DINOV3_WEIGHTS_SHA256" ]]; then
    echo "DINOv3 artifact SHA-256 mismatch; refusing to use downloaded weights" >&2
    exit 1
  fi
  mv -f "$temporary_weights" "$weights_path"
  trap - EXIT
fi
printf '%s  %s\n' "$DINOV3_WEIGHTS_SHA256" "$DINOV3_WEIGHTS_FILE" > .artifacts/dinov3/weights.sha256
