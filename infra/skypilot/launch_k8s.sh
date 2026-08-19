#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 MODE --workspace NAME [volume options] [artifact options]" >&2
  echo "       [--run-id ID [--resume]] [--distributed-smoke-run-id ID]" >&2
  echo "       [--qualification-run-id ID] [--runtime-readiness-run-id ID]" >&2
  echo "       [--git-url URL --git-ref COMMIT] [--priority p0|p1|p2|p3|p4] [--dry-run]" >&2
  echo "MODE: preflight | stage | stage-prefill | stage-dinov3 | stage-released | distributed-smoke | qualify | train-smoke | full" >&2
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
sky_executable="${SKY_EXECUTABLE:-$HOME/.sky/bin/sky}"
python_for_sky="${SKY_PYTHON:-$HOME/.sky/bin/sky-python/bin/python}"

mode="${1:-}"
[[ -n "$mode" ]] || { usage; exit 2; }
shift

droid_volume=""
checkpoint_volume=""
dinov3_weights_sha256=""
run_id=""
qualification_run_id=""
distributed_smoke_run_id=""
runtime_readiness_run_id=""
resume_requested=0
dry_run=0
git_url=""
git_ref=""
sky_workspace=""
k8s_context="Skypilot"
priority_class="p1"
while (($#)); do
  case "$1" in
    --droid-volume) droid_volume="${2:-}"; shift 2 ;;
    --checkpoint-volume) checkpoint_volume="${2:-}"; shift 2 ;;
    --dinov3-weights-sha256) dinov3_weights_sha256="${2:-}"; shift 2 ;;
    --run-id) run_id="${2:-}"; shift 2 ;;
    --qualification-run-id) qualification_run_id="${2:-}"; shift 2 ;;
    --distributed-smoke-run-id) distributed_smoke_run_id="${2:-}"; shift 2 ;;
    --runtime-readiness-run-id) runtime_readiness_run_id="${2:-}"; shift 2 ;;
    --resume) resume_requested=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    --git-url) git_url="${2:-}"; shift 2 ;;
    --git-ref) git_ref="${2:-}"; shift 2 ;;
    --workspace) sky_workspace="${2:-}"; shift 2 ;;
    --k8s-context) k8s_context="${2:-}"; shift 2 ;;
    --priority) priority_class="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

validate_volume_name() {
  local value="$1"
  local label="$2"
  if [[ ! "$value" =~ ^[a-z0-9]([-a-z0-9.]{0,61}[a-z0-9])?$ ]]; then
    echo "$label must be a lowercase DNS-style Sky volume name (max 63 characters)" >&2
    exit 2
  fi
}

validate_dinov3_sha() {
  dinov3_weights_sha256="$(printf '%s' "$dinov3_weights_sha256" | tr '[:upper:]' '[:lower:]')"
  if [[ ! "$dinov3_weights_sha256" =~ ^8aa4cbdd[[:xdigit:]]{56}$ ]]; then
    echo "The native DINOv3 ViT-L/16 full SHA-256 (8aa4cbdd... identity) is required" >&2
    exit 2
  fi
}

if [[ -n "$run_id" ]] && [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]]; then
  echo "--run-id must be 1-128 filesystem-safe characters" >&2
  exit 2
fi
if [[ -n "$qualification_run_id" ]] && [[ ! "$qualification_run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]]; then
  echo "--qualification-run-id must be 1-128 filesystem-safe characters" >&2
  exit 2
fi
if [[ -n "$distributed_smoke_run_id" ]] && [[ ! "$distributed_smoke_run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]]; then
  echo "--distributed-smoke-run-id must be 1-128 filesystem-safe characters" >&2
  exit 2
fi
if [[ -n "$runtime_readiness_run_id" ]] && [[ ! "$runtime_readiness_run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]]; then
  echo "--runtime-readiness-run-id must be 1-128 filesystem-safe characters" >&2
  exit 2
fi
if ((resume_requested)) && [[ -z "$run_id" ]]; then
  echo "--resume requires an explicit --run-id" >&2
  exit 2
fi

cd "$repo_root"
[[ -x "$python_for_sky" ]] || python_for_sky="$(command -v python3)"
"$python_for_sky" infra/skypilot/preflight.py --quiet

if [[ "$mode" == preflight ]]; then
  echo "Kubernetes preflight passed; no job or volume was created."
  exit 0
fi

case "$mode" in
  stage|stage-prefill)
    task_spec=infra/skypilot/droid_stage.yaml
    [[ -n "$droid_volume" ]] || { echo "--droid-volume is required" >&2; exit 2; }
    ;;
  stage-dinov3)
    task_spec=infra/skypilot/dinov3_stage_k8s.yaml
    [[ -n "$checkpoint_volume" ]] || { echo "--checkpoint-volume is required" >&2; exit 2; }
    validate_dinov3_sha
    ;;
  stage-released)
    task_spec=infra/skypilot/released_droid_stage_k8s.yaml
    [[ -n "$checkpoint_volume" ]] || { echo "--checkpoint-volume is required" >&2; exit 2; }
    ;;
  distributed-smoke)
    task_spec=infra/skypilot/distributed_smoke.yaml
    [[ -n "$checkpoint_volume" ]] || { echo "--checkpoint-volume is required" >&2; exit 2; }
    ;;
  qualify|train-smoke|full)
    case "$mode" in
      qualify) task_spec=infra/skypilot/droid_qualify_released.yaml ;;
      train-smoke) task_spec=infra/skypilot/droid_train_smoke.yaml ;;
      full) task_spec=infra/skypilot/droid_train_full.yaml ;;
    esac
    [[ -n "$droid_volume" ]] || { echo "--droid-volume is required" >&2; exit 2; }
    [[ -n "$checkpoint_volume" ]] || { echo "--checkpoint-volume is required" >&2; exit 2; }
    validate_dinov3_sha
    if [[ "$mode" == train-smoke && -z "$distributed_smoke_run_id" ]]; then
      echo "train-smoke requires --distributed-smoke-run-id from the completed cross-node smoke" >&2
      exit 2
    fi
    if [[ "$mode" == full && -z "$qualification_run_id" ]]; then
      echo "full requires --qualification-run-id from a completed released-checkpoint qualification" >&2
      exit 2
    fi
    if [[ "$mode" == full && -z "$runtime_readiness_run_id" ]]; then
      echo "full requires --runtime-readiness-run-id from the completed real training smoke" >&2
      exit 2
    fi
    ;;
  *) echo "Unknown mode: $mode" >&2; usage; exit 2 ;;
esac

[[ -z "$droid_volume" ]] || validate_volume_name "$droid_volume" "--droid-volume"
[[ -z "$checkpoint_volume" ]] || validate_volume_name "$checkpoint_volume" "--checkpoint-volume"
if [[ ! "$k8s_context" =~ ^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$ ]]; then
  echo "--k8s-context is invalid" >&2
  exit 2
fi
if [[ ! "$priority_class" =~ ^p[0-4]$ ]]; then
  echo "--priority must be one of the case-sensitive Enterprise classes p0, p1, p2, p3, or p4" >&2
  exit 2
fi

rendered_task="$(mktemp "${TMPDIR:-/tmp}/jepawm-k8s-task.XXXXXX.yaml")"
trap 'rm -f "$rendered_task"' EXIT
render_args=(--input "$task_spec" --output "$rendered_task" --context "$k8s_context")
[[ -z "$droid_volume" ]] || render_args+=(--droid-volume "$droid_volume")
[[ -z "$checkpoint_volume" ]] || render_args+=(--checkpoint-volume "$checkpoint_volume")
"$python_for_sky" infra/skypilot/render_k8s_task.py "${render_args[@]}"
"$python_for_sky" - "$rendered_task" <<'PY'
import sys
import sky

sky.Task.from_yaml(sys.argv[1])
PY

if ((dry_run)); then
  echo "Kubernetes profile for '$mode' rendered and passed Sky schema validation; no job was submitted."
  exit 0
fi

if [[ ! "$git_url" =~ ^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(\.git)?$ ]]; then
  echo "Actual launches require --git-url https://github.com/OWNER/REPO[.git]" >&2
  exit 2
fi
git_ref="$(printf '%s' "$git_ref" | tr '[:upper:]' '[:lower:]')"
if [[ ! "$git_ref" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Actual launches require an immutable 40-hex --git-ref" >&2
  exit 2
fi
if [[ -z "$sky_workspace" ]] || [[ "$sky_workspace" == *[$' \t\r\n']* ]]; then
  echo "Actual launches require a whitespace-free --workspace" >&2
  exit 2
fi

if [[ ! -x "$sky_executable" ]]; then
  sky_executable="$(command -v sky)"
fi
availability="$($sky_executable check -w "$sky_workspace" -o json)"
python3 - "$sky_workspace" "$availability" <<'PY'
import json
import sys

workspace = sys.argv[1]
availability = json.loads(sys.argv[2])
capabilities = availability.get(workspace, {})
if "Kubernetes" not in capabilities or "compute" not in capabilities["Kubernetes"]:
    raise SystemExit(f"Sky workspace {workspace!r} has no Kubernetes compute capability")
PY

launch_env=()
[[ "$mode" != stage-prefill ]] || launch_env+=(--env "JEPAWM_STAGE_PREFILL_ONLY=1")
[[ -z "$dinov3_weights_sha256" ]] || launch_env+=(--env "DINOV3_WEIGHTS_SHA256=$dinov3_weights_sha256")
[[ -z "$run_id" ]] || launch_env+=(--env "JEPAWM_RUN_ID=$run_id")
[[ -z "$qualification_run_id" ]] || launch_env+=(--env "JEPAWM_QUALIFICATION_RUN_ID=$qualification_run_id")
[[ -z "$distributed_smoke_run_id" ]] || launch_env+=(--env "JEPAWM_DISTRIBUTED_SMOKE_RUN_ID=$distributed_smoke_run_id")
[[ -z "$runtime_readiness_run_id" ]] || launch_env+=(--env "JEPAWM_RUNTIME_READINESS_RUN_ID=$runtime_readiness_run_id")
((resume_requested == 0)) || launch_env+=(--env "JEPAWM_RESUME=1")

# Use p1 first so this run can displace p2 work without taking p0 capacity.
# Callers may explicitly escalate to p0 if p1 cannot secure enough H200s.
launch_command=(
  "$sky_executable" jobs launch "$rendered_task" --priority "$priority_class" -y -d
  --git-url "$git_url" --git-ref "$git_ref" --workspace "$sky_workspace"
)
# macOS ships Bash 3.2, where expanding an empty array under `set -u` raises
# "unbound variable".  Stage jobs legitimately have no launch-time env
# overrides, so append this array only when it is non-empty.
if ((${#launch_env[@]})); then
  launch_command+=("${launch_env[@]}")
fi
"${launch_command[@]}"
