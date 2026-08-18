#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 MODE [stores] [--dinov3-weights-uri gs://URI --dinov3-weights-sha256 HEX]" >&2
  echo "       [--run-id ID [--resume]] [--distributed-smoke-run-id ID]" >&2
  echo "       [--qualification-run-id ID] [--runtime-readiness-run-id ID]" >&2
  echo "       [--git-url URL --git-ref COMMIT --workspace NAME] [--dry-run]" >&2
  echo "MODE: preflight | stage | distributed-smoke | train-smoke | qualify | full" >&2
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
sky_executable="${SKY_EXECUTABLE:-$HOME/.sky/bin/sky}"

mode="${1:-}"
if [[ -z "$mode" ]]; then
  usage
  exit 2
fi
shift

checkpoint_store=""
droid_store=""
dinov3_weights_uri=""
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
while (($#)); do
  case "$1" in
    --checkpoint-store)
      checkpoint_store="${2:-}"
      shift 2
      ;;
    --droid-store)
      droid_store="${2:-}"
      shift 2
      ;;
    --dinov3-weights-uri)
      dinov3_weights_uri="${2:-}"
      shift 2
      ;;
    --dinov3-weights-sha256)
      dinov3_weights_sha256="${2:-}"
      shift 2
      ;;
    --run-id)
      run_id="${2:-}"
      shift 2
      ;;
    --qualification-run-id)
      qualification_run_id="${2:-}"
      shift 2
      ;;
    --distributed-smoke-run-id)
      distributed_smoke_run_id="${2:-}"
      shift 2
      ;;
    --runtime-readiness-run-id)
      runtime_readiness_run_id="${2:-}"
      shift 2
      ;;
    --resume)
      resume_requested=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --git-url)
      git_url="${2:-}"
      shift 2
      ;;
    --git-ref)
      git_ref="${2:-}"
      shift 2
      ;;
    --workspace)
      sky_workspace="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

validate_checkpoint_uri() {
  local value="$1"
  if [[ "$value" == *[$' \t\r\n']* ]]; then
    echo "Checkpoint storage URI must not contain whitespace" >&2
    return 1
  fi
  case "$value" in
    gs://*|s3://*|r2://*|hf://buckets/*|nebius://*|cw://*|oci://*|vastdata://*|minio://*) ;;
    *)
      echo "Unsupported checkpoint storage URI: use a SkyPilot object-store URI" >&2
      return 1
      ;;
  esac
}

validate_droid_uri() {
  local value="$1"
  if [[ "$value" != gs://* ]] || [[ "$value" == *[$' \t\r\n']* ]]; then
    echo "DROID storage must be a whitespace-free gs:// bucket or prefix" >&2
    return 1
  fi
}

validate_dinov3_artifact() {
  if [[ "$dinov3_weights_uri" != gs://* ]] || [[ "$dinov3_weights_uri" == *[$' \t\r\n']* ]]; then
    echo "DINOv3 weights must use a non-secret, whitespace-free gs:// artifact URI" >&2
    return 1
  fi
  dinov3_weights_sha256="$(printf '%s' "$dinov3_weights_sha256" | tr '[:upper:]' '[:lower:]')"
  if [[ ! "$dinov3_weights_sha256" =~ ^8aa4cbdd[[:xdigit:]]{56}$ ]]; then
    echo "DINOv3 weights require the full SHA-256 matching the official ViT-L/16 8aa4cbdd .pth identity" >&2
    return 1
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
  echo "--resume requires an explicit --run-id for intentional continuation" >&2
  exit 2
fi

cd "$repo_root"
python_for_sky="${SKY_PYTHON:-$HOME/.sky/bin/sky-python/bin/python}"
if [[ ! -x "$python_for_sky" ]]; then
  python_for_sky="$(command -v python3)"
fi
"$python_for_sky" infra/skypilot/preflight.py --quiet

if [[ "$mode" == "preflight" ]]; then
  echo "Preflight passed; no job was submitted."
  exit 0
fi

case "$mode" in
  stage)
    task_spec=infra/skypilot/droid_stage.yaml
    [[ -n "$droid_store" ]] || { echo "--droid-store is required" >&2; exit 2; }
    validate_droid_uri "$droid_store"
    launch_env=(--env "DROID_STORE_URI=$droid_store")
    ;;
  distributed-smoke)
    task_spec=infra/skypilot/distributed_smoke.yaml
    [[ -n "$checkpoint_store" ]] || { echo "--checkpoint-store is required" >&2; exit 2; }
    validate_checkpoint_uri "$checkpoint_store"
    launch_env=(--env "CHECKPOINT_STORE_URI=$checkpoint_store")
    ;;
  train-smoke)
    task_spec=infra/skypilot/droid_train_smoke.yaml
    ;;
  qualify)
    task_spec=infra/skypilot/droid_qualify_released.yaml
    ;;
  full)
    task_spec=infra/skypilot/droid_train_full.yaml
    ;;
  *)
    echo "Unknown mode: $mode" >&2
    usage
    exit 2
    ;;
esac

if [[ "$mode" == "train-smoke" || "$mode" == "qualify" || "$mode" == "full" ]]; then
  [[ -n "$checkpoint_store" ]] || { echo "--checkpoint-store is required" >&2; exit 2; }
  [[ -n "$droid_store" ]] || { echo "--droid-store is required" >&2; exit 2; }
  validate_checkpoint_uri "$checkpoint_store"
  validate_droid_uri "$droid_store"
  [[ -n "$dinov3_weights_uri" ]] || { echo "--dinov3-weights-uri is required" >&2; exit 2; }
  [[ -n "$dinov3_weights_sha256" ]] || { echo "--dinov3-weights-sha256 is required" >&2; exit 2; }
  validate_dinov3_artifact
  launch_env=(
    --env "CHECKPOINT_STORE_URI=$checkpoint_store"
    --env "DROID_STORE_URI=$droid_store"
    --env "DINOV3_WEIGHTS_URI=$dinov3_weights_uri"
    --env "DINOV3_WEIGHTS_SHA256=$dinov3_weights_sha256"
  )
  if [[ -n "$run_id" ]]; then
    launch_env+=(--env "JEPAWM_RUN_ID=$run_id")
  fi
  if ((resume_requested)); then
    launch_env+=(--env "JEPAWM_RESUME=1")
  fi
  if [[ "$mode" == "train-smoke" ]]; then
    [[ -n "$distributed_smoke_run_id" ]] || {
      echo "train-smoke requires --distributed-smoke-run-id from the completed cross-node smoke" >&2
      exit 2
    }
    launch_env+=(--env "JEPAWM_DISTRIBUTED_SMOKE_RUN_ID=$distributed_smoke_run_id")
  fi
  if [[ "$mode" == "full" ]]; then
    [[ -n "$qualification_run_id" ]] || {
      echo "full requires --qualification-run-id from a completed released-checkpoint qualification" >&2
      exit 2
    }
    launch_env+=(--env "JEPAWM_QUALIFICATION_RUN_ID=$qualification_run_id")
    [[ -n "$runtime_readiness_run_id" ]] || {
      echo "full requires --runtime-readiness-run-id from the completed real training smoke" >&2
      exit 2
    }
    launch_env+=(--env "JEPAWM_RUNTIME_READINESS_RUN_ID=$runtime_readiness_run_id")
  fi
fi

if [[ "$mode" == "distributed-smoke" && -n "$run_id" ]]; then
  launch_env+=(--env "JEPAWM_RUN_ID=$run_id")
fi

if ((dry_run)); then
  echo "Preflight passed for mode '$mode'; no job was submitted."
  exit 0
fi

if [[ ! "$git_url" =~ ^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(\.git)?$ ]]; then
  echo "Actual launches require --git-url https://github.com/OWNER/REPO[.git]" >&2
  exit 2
fi
git_ref="$(printf '%s' "$git_ref" | tr '[:upper:]' '[:lower:]')"
if [[ ! "$git_ref" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Actual launches require --git-ref with the exact 40-hex commit, not a moving branch or tag" >&2
  exit 2
fi
if [[ -z "$sky_workspace" ]] || [[ "$sky_workspace" == *[$' \t\r\n']* ]]; then
  echo "Actual launches require a whitespace-free --workspace with GCP enabled" >&2
  exit 2
fi

if [[ ! -x "$sky_executable" ]]; then
  sky_executable="$(command -v sky)"
fi

# P1 intentionally remains a launch-time priority class rather than being
# hidden in task YAML.  This Enterprise workspace defines class names in
# lowercase, so `p1` is the exact accepted spelling of the P1 class.
exec "$sky_executable" jobs launch "$task_spec" --priority p1 \
  --git-url "$git_url" --git-ref "$git_ref" --workspace "$sky_workspace" \
  "${launch_env[@]}"
