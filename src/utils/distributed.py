# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import datetime
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.elastic.utils.distributed import get_free_port

from src.utils.logging import get_logger

logger = get_logger()


def _get_port(world_size, default_port=37129):
    # If other jobs are running on the node, the default_port might be in use by another process. If we are
    # only using 1 GPU, we can avoid this by just picking a free port
    return get_free_port() if world_size == 1 else default_port


def _configure_process_group_environment(
    port=None,
    rank_and_world_size=(None, None),
):
    """Resolve rank/rendezvous variables without clobbering a launcher.

    ``torchrun`` owns ``MASTER_ADDR`` and ``MASTER_PORT``.  In particular, a
    multi-node launch passes the head node address through those variables;
    replacing either value with a per-node default partitions the job into
    independent, hanging process groups.

    Returns:
        ``(world_size, rank, should_initialize)``.  The last element is false
        only for the non-distributed fallback.
    """

    rank, world_size = rank_and_world_size
    torchrun_keys = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    torchrun_present = [key in os.environ for key in torchrun_keys]
    if any(torchrun_present) and not all(torchrun_present):
        missing = [key for key, present in zip(torchrun_keys, torchrun_present) if not present]
        raise RuntimeError(f"Incomplete torchrun environment; missing: {', '.join(missing)}")

    if all(torchrun_present):
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
    elif (rank is not None) and (world_size is not None):
        # Compatibility with the legacy local multiprocessing launcher.
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ.setdefault("LOCAL_RANK", str(rank))
    else:
        try:
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            world_size = int(os.environ["WORLD_SIZE"])
            rank = int(os.environ["RANK"])
            # Submitit may already provide the correct head-node address.  A
            # hostname fallback is retained for older single-node setups.
            os.environ.setdefault(
                "MASTER_ADDR",
                os.environ.get("SLURM_LAUNCH_NODE_IPADDR", os.environ.get("HOSTNAME", socket.gethostname())),
            )
        except KeyError as exc:
            logger.info(f"SLURM vars not set (distributed training not available): {exc}")
            return 1, 0, False

    world_size = int(world_size)
    rank = int(rank)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    if port is not None:
        os.environ["MASTER_PORT"] = str(port)
    else:
        os.environ.setdefault("MASTER_PORT", str(_get_port(world_size)))
    return world_size, rank, True


def init_distributed(
    port=None,
    rank_and_world_size=(None, None),
    nccl_timeout_minutes=None,
):
    # Set all environment variables *before* calling `torch.distributed.init_process_group`. `init_process_group` may
    # reallocate environment variables; modifying them after could trigger a race condition leading to a segfault.
    if "SLURM_JOB_ID" in os.environ:
        # Use the slurm_tmpdir (if it exists) instead of /tmp
        tmpdir = Path(f"/scratch/slurm_tmpdir/{os.environ['SLURM_JOB_ID']}")
        if tmpdir.exists():
            os.environ["TMPDIR"] = str(tmpdir)

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    world_size, rank, should_initialize = _configure_process_group_environment(
        port=port,
        rank_and_world_size=rank_and_world_size,
    )
    if not should_initialize:
        return world_size, rank

    try:
        # Increase timeout for large-scale multi-node jobs
        # Also need longer timeout for mixed video+image training where different loaders
        # (e.g., Instagram video loader) can take a very long time to fetch the first sample
        nccl_timeout = None
        if nccl_timeout_minutes is not None:
            logger.info(f"Initializing distributed with timeout={nccl_timeout_minutes} minutes")
            nccl_timeout = datetime.timedelta(minutes=nccl_timeout_minutes)
        torch.distributed.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            world_size=world_size,
            rank=rank,
            timeout=nccl_timeout,
        )
    except Exception as e:
        logger.error(f"Rank {rank}: Distributed training initialization FAILED: {e}")
        logger.error("This is a fatal error for multi-GPU training. Check network connectivity.")
        # Re-raise the exception instead of silently continuing with world_size=1
        # This prevents confusing errors later when DDP fails
        raise RuntimeError(
            f"Failed to initialize distributed training: {e}. "
            f"Rank={rank}, World={world_size}, Master={os.environ.get('MASTER_ADDR')}"
        ) from e

    return world_size, rank


def is_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    for key in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]:
        if key not in os.environ:
            return False
    return True


def get_local_rank() -> int:
    assert is_initialized()
    return int(os.environ["LOCAL_RANK"])


def get_global_rank() -> int:
    assert is_initialized()
    return int(os.environ["RANK"])


def get_world_size() -> int:
    assert is_initialized()
    return int(os.environ["WORLD_SIZE"])


class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
