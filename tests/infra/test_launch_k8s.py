import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_stage_launch_accepts_no_optional_environment_overrides(tmp_path: Path):
    invocation_log = tmp_path / "sky-invocation.txt"
    fake_sky = tmp_path / "sky"
    fake_sky.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == check ]]; then
  printf '{"default":{"Kubernetes":["compute"]}}\\n'
  exit 0
fi
printf '%s\\n' "$@" > "$JEPAWM_TEST_SKY_LOG"
""",
        encoding="utf-8",
    )
    fake_sky.chmod(0o755)

    # The wrapper uses Python for local preflight, task rendering/schema
    # validation, and workspace-capability parsing.  Those components have
    # dedicated tests; this stub isolates the Bash 3.2 empty-array path.
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "SKY_EXECUTABLE": str(fake_sky),
            "SKY_PYTHON": str(fake_python),
            "JEPAWM_TEST_SKY_LOG": str(invocation_log),
        }
    )
    subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "infra/skypilot/launch_k8s.sh"),
            "stage",
            "--droid-volume",
            "firstuserhere-jepawm-droid",
            "--git-url",
            "https://github.com/firstuserhere/jepa-wms.git",
            "--git-ref",
            "f1e6237a74b99d007cac6590deabce087ea00059",
            "--workspace",
            "default",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
    )

    arguments = invocation_log.read_text(encoding="utf-8").splitlines()
    assert arguments[:2] == ["jobs", "launch"]
    assert "--priority" in arguments
    assert "p3" in arguments
    assert "-y" in arguments
    assert "-d" in arguments
    assert "--env" not in arguments

    subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "infra/skypilot/launch_k8s.sh"),
            "stage",
            "--droid-volume",
            "firstuserhere-jepawm-droid",
            "--priority",
            "p0",
            "--git-url",
            "https://github.com/firstuserhere/jepa-wms.git",
            "--git-ref",
            "f1e6237a74b99d007cac6590deabce087ea00059",
            "--workspace",
            "default",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
    )
    escalated_arguments = invocation_log.read_text(encoding="utf-8").splitlines()
    priority_index = escalated_arguments.index("--priority")
    assert escalated_arguments[priority_index + 1] == "p0"


def test_stage_prefill_is_explicit_and_fail_closed(tmp_path: Path):
    invocation_log = tmp_path / "sky-invocation.txt"
    fake_sky = tmp_path / "sky"
    fake_sky.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == check ]]; then
  printf '{"default":{"Kubernetes":["compute"]}}\\n'
  exit 0
fi
printf '%s\\n' "$@" > "$JEPAWM_TEST_SKY_LOG"
""",
        encoding="utf-8",
    )
    fake_sky.chmod(0o755)
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "SKY_EXECUTABLE": str(fake_sky),
            "SKY_PYTHON": str(fake_python),
            "JEPAWM_TEST_SKY_LOG": str(invocation_log),
        }
    )

    subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "infra/skypilot/launch_k8s.sh"),
            "stage-prefill",
            "--droid-volume",
            "firstuserhere-jepawm-droid",
            "--git-url",
            "https://github.com/firstuserhere/jepa-wms.git",
            "--git-ref",
            "f1e6237a74b99d007cac6590deabce087ea00059",
            "--workspace",
            "default",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
    )

    arguments = invocation_log.read_text(encoding="utf-8").splitlines()
    env_index = arguments.index("--env")
    assert arguments[env_index + 1] == "JEPAWM_STAGE_PREFILL_ONLY=1"
