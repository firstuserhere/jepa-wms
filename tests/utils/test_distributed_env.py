import os

import pytest

from src.utils.distributed import _configure_process_group_environment

DIST_ENV_KEYS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "SLURM_NTASKS",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_LAUNCH_NODE_IPADDR",
)


@pytest.fixture(autouse=True)
def clean_distributed_environment(monkeypatch):
    for key in DIST_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_torchrun_rendezvous_is_preserved(monkeypatch):
    monkeypatch.setenv("RANK", "19")
    monkeypatch.setenv("WORLD_SIZE", "32")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.8")
    monkeypatch.setenv("MASTER_PORT", "29417")

    world_size, rank, should_initialize = _configure_process_group_environment()

    assert (world_size, rank, should_initialize) == (32, 19, True)
    assert os.environ["MASTER_ADDR"] == "10.0.0.8"
    assert os.environ["MASTER_PORT"] == "29417"
    assert os.environ["LOCAL_RANK"] == "3"


def test_explicit_port_is_the_only_way_to_override_torchrun_port(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "head.internal")
    monkeypatch.setenv("MASTER_PORT", "29417")

    _configure_process_group_environment(port=31234)

    assert os.environ["MASTER_ADDR"] == "head.internal"
    assert os.environ["MASTER_PORT"] == "31234"


def test_partial_torchrun_environment_fails_closed(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")

    with pytest.raises(RuntimeError, match="LOCAL_RANK"):
        _configure_process_group_environment()


def test_legacy_local_launcher_gets_local_rendezvous():
    world_size, rank, should_initialize = _configure_process_group_environment(
        port=32100,
        rank_and_world_size=(2, 8),
    )

    assert (world_size, rank, should_initialize) == (8, 2, True)
    assert os.environ["LOCAL_RANK"] == "2"
    assert os.environ["MASTER_ADDR"] == "localhost"
    assert os.environ["MASTER_PORT"] == "32100"


def test_slurm_preserves_supplied_head_address(monkeypatch):
    monkeypatch.setenv("SLURM_NTASKS", "16")
    monkeypatch.setenv("SLURM_PROCID", "9")
    monkeypatch.setenv("SLURM_LOCALID", "1")
    monkeypatch.setenv("MASTER_ADDR", "slurm-head.internal")
    monkeypatch.setenv("MASTER_PORT", "29876")

    world_size, rank, should_initialize = _configure_process_group_environment()

    assert (world_size, rank, should_initialize) == (16, 9, True)
    assert os.environ["MASTER_ADDR"] == "slurm-head.internal"
    assert os.environ["MASTER_PORT"] == "29876"
