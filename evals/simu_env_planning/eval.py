# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import logging
import os
import tempfile
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from omegaconf import OmegaConf

from evals.simu_env_planning.envs.init import make_env
from evals.simu_env_planning.planning.common.gc_logger import Logger
from evals.simu_env_planning.planning.common.parser import parse_cfg
from evals.simu_env_planning.planning.gc_agent import GC_Agent
from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
from evals.simu_env_planning.planning.utils import aggregate_results, compute_task_distribution, set_seed
from evals.utils import make_datasets
from src.utils.planning_promotion import (
    build_complete_planning_result,
    validate_planning_provenance,
    verify_planning_checkpoint,
    write_complete_planning_result,
)
from src.utils.yaml_utils import expand_env_vars

# Submitit/Slurm tasks are pinned to one visible device.  torchrun processes
# keep all devices visible and bind from LOCAL_RANK below.
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

from src.utils.distributed import init_distributed

# ------------------------------

logging.basicConfig()
log = logging.getLogger()

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


def _atomic_dump_yaml(path, payload):
    """Atomically publish a YAML config from rank zero."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as yaml_file:
            yaml.dump(payload, yaml_file, default_flow_style=False)
            yaml_file.flush()
            os.fsync(yaml_file.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _checkpoint_path(checkpoint_folder, checkpoint):
    checkpoint_path = Path(checkpoint).expanduser()
    if checkpoint_path.is_absolute():
        return checkpoint_path
    return Path(checkpoint_folder).expanduser() / checkpoint_path


def _episode_counts(rank_results, tasks):
    """Count episodes once per task (every metric tuple carries the count)."""

    counts = {str(task): 0 for task in tasks}
    for results in rank_results:
        for task in tasks:
            task = str(task)
            canonical_key = f"ep_end_dist+{task}"
            fallback_key = f"episode_reward+{task}"
            value = results.get(canonical_key, results.get(fallback_key))
            if value is not None:
                counts[task] += int(value[1])
    return counts


def main(args_eval, resume_preempt=False):

    # Expand environment variables in the config
    args_eval = expand_env_vars(args_eval)

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #
    eval_tag = args_eval.get("tag", None)
    # -- PRETRAIN
    args_pretrain = args_eval.get("model_kwargs")
    module_name = args_pretrain.get("module_name")
    pretrain_folder = args_eval.get("folder", None)
    checkpoint_folder = args_eval.get("checkpoint_folder", pretrain_folder) or pretrain_folder
    checkpoint = args_pretrain.get("checkpoint")
    planning_provenance = args_eval.get("planning_provenance")
    if planning_provenance is not None:
        # Validate before adding derived runtime keys such as work_dir.
        planning_provenance = validate_planning_provenance(planning_provenance, eval_config=args_eval)

    # -- log/checkpointing paths
    folder = os.path.join(pretrain_folder, "simu_env_planning/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    # -- Distributed
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        # Submitit pins each Slurm task to one visible GPU.  torchrun leaves all
        # local GPUs visible and communicates the binding through LOCAL_RANK.
        slurm_pinned = "SLURM_LOCALID" in os.environ
        visible_devices = [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item]
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_index = 0 if slurm_pinned or len(visible_devices) == 1 else local_rank
        device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(device)
    world_size, rank = init_distributed()
    log.info(f"🚀 Initialized (rank/world-size) {rank}/{world_size}")

    # One writer prevents truncated configs when every distributed process
    # starts simultaneously on the same durable filesystem.
    yaml_file_path = os.path.join(folder, "args_eval.yaml")
    config_write = [None]
    if rank == 0:
        try:
            _atomic_dump_yaml(yaml_file_path, args_eval)
            log.info(f"📁 Saved args_eval to {yaml_file_path}")
        except Exception as error:
            config_write[0] = f"{type(error).__name__}: {error}"
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(config_write, src=0)
    if config_write[0] is not None:
        raise RuntimeError(f"could not persist planning evaluation config: {config_write[0]}")

    if planning_provenance is not None:
        verification = [None]
        if rank == 0:
            try:
                immutable_path = _checkpoint_path(checkpoint_folder, checkpoint)
                verify_planning_checkpoint(planning_provenance, immutable_path)
                log.info(
                    "🔒 Verified immutable planning checkpoint %s (%s)",
                    planning_provenance["checkpoint_id"],
                    planning_provenance["checkpoint_sha256"],
                )
            except Exception as error:
                verification[0] = f"{type(error).__name__}: {error}"
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(verification, src=0)
        if verification[0] is not None:
            raise RuntimeError(f"planning checkpoint provenance verification failed: {verification[0]}")

    model_kwargs = args_eval["model_kwargs"]

    # -- Initialize model
    if importlib.util.find_spec(module_name) is None:
        raise NotImplementedError(f"Module {module_name} not found")
    cfgs_data = model_kwargs.get("data", {})
    cfgs_data_aug = model_kwargs.get("data_aug", {})
    wrapper_kwargs = model_kwargs.get("wrapper_kwargs", {})
    pretrain_kwargs = model_kwargs.get("pretrain_kwargs", {})
    dset, preprocessor = make_datasets(cfgs_data, cfgs_data_aug, world_size, rank)
    args_eval["frameskip"] = cfgs_data["custom"]["frameskip"]
    args_eval["work_dir"] = folder
    model = init_module(
        folder=checkpoint_folder,
        checkpoint=checkpoint,
        module_name=module_name,
        model_kwargs=pretrain_kwargs,
        wrapper_kwargs=wrapper_kwargs,
        cfgs_data=cfgs_data,
        device=device,
        action_dim=dset.action_dim,
        proprio_dim=dset.proprio_dim,
        preprocessor=preprocessor,
    )
    log.info("✅ Loaded encoder and predictor")

    # -- Launch eval
    main_distributed_episodes_eval(args_eval, model=model, dset=dset, preprocessor=preprocessor, rank=rank)


def main_distributed_episodes_eval(
    cfg: dict,
    model=None,
    dset=None,
    preprocessor=None,
    rank=0,
    device="cuda:0",
    process_group=None,
):
    """
    Should work with one or more GPUs, even with distribute_multitask_eval=False.
    If world_size > 1 and distribute_multitask_eval=False, will all have same task_indices
    and potentially different results if we have a seed_shift in local_rngs but only rank 0
    will be logged. world_size=1 and distribute_multitask_eval=True should also work.
    To avoid timeout when gathering results from all ranks we create dummy episodes, The logic
    raises an assertion error in case the total_evals_episodes = tasks * eval_episodes < world_size.
    """
    # Setup the config
    start_time = time()
    cfg = OmegaConf.create(cfg)
    planning_provenance = cfg.get("planning_provenance")
    if planning_provenance is not None:
        plain_cfg = OmegaConf.to_container(cfg, resolve=True)
        planning_provenance = validate_planning_provenance(
            OmegaConf.to_container(planning_provenance, resolve=True), eval_config=plain_cfg
        )
    cfg = parse_cfg(cfg)
    set_seed(cfg.meta.seed)
    cfg.rank = rank
    # A training job can evaluate on a prefix subgroup (for example the exact
    # released DROID 1-node/8-rank topology) while the remaining training ranks
    # wait at an outer synchronization point.  All collectives in this routine
    # must therefore remain scoped to this group.
    cfg.world_size = dist.get_world_size(group=process_group)
    cfg.device = device
    cfg.num_active_gpus = cfg.world_size
    cfg.active_ranks = [i for i in range(cfg.world_size)]
    log.info(f"{cfg.active_ranks=}")
    if cfg.rank == 0:
        log.info(f"📂 Work dir: {cfg.work_dir}")
    cfg.task_specification.goal_source = cfg.task_specification.get("goal_source", "expert")
    # DEFINE cfg.action_ratio := simu_actions / wm_fw_passes
    # TODO, replace all mentions of cfg.frameskip by cfg.action_ratio in the logging logic
    # of this script
    if cfg.planner.repeat_actskip:
        cfg.action_ratio = 1
    else:
        cfg.action_ratio = cfg.frameskip // model.action_skip
    log.info("First env creation just to define cfg.action_dim")
    env = make_env(cfg)  # needed here to define cfg.action_dim

    # The per-episode barrier requires every rank to execute the same positive
    # number of episodes.  Padding handles uneven division, but cannot pad a
    # rank that was assigned no task at all.
    total_eval_episodes = int(cfg.meta.eval_episodes) * len(cfg.tasks)
    if cfg.distributed.distribute_multitask_eval and total_eval_episodes < cfg.world_size:
        raise ValueError(
            "distributed planning evaluation requires at least one episode per rank: "
            f"{total_eval_episodes} episodes for {cfg.world_size} ranks"
        )

    cfg.planner.distribute_planner = False
    cfg.local_seed = cfg.meta.seed
    if cfg.distributed.distribute_multitask_eval:
        # Set a unique seed for the sampler based on the process rank
        if cfg.distributed.seed_shift == "horizon_1000":
            seed_shift = cfg.planner.horizon * 1000
        else:
            if isinstance(cfg.distributed.seed_shift, int) or isinstance(cfg.distributed.seed_shift, float):
                seed_shift = cfg.distributed.seed_shift
            else:
                raise ValueError("cfg.distributed.seed_shift does not have correct format")
        # We do not want to put local rng samplers in mujoco envs so put a different
        # global seed for each process, to ensure independence of environments
        cfg.local_seed += cfg.rank * seed_shift
        if not cfg.distributed.local_rng_samplers:
            set_seed(cfg.local_seed)
            log.info(f"Local Seed={cfg.local_seed} set for entire eval for rank {cfg.rank}")
        else:
            # In gc_planning will be used to seed envs
            log.info(f"Initialized local rng with seed {cfg.local_seed} for rank {cfg.rank}")

    if cfg.meta.quick_debug:
        log.info("Quick debug mode enabled.")
        cfg.meta.eval_episodes = 1
        cfg.planner.iterations = 2
        cfg.planner.num_samples = 2
        cfg.planner.num_elites = 2
        cfg.logging.tqdm_silent = False
    if cfg.planner.planner_name in ["cem", "mppi", "nevergrad"]:
        assert cfg.planner.num_elites <= cfg.planner.num_samples, "num_elites should be <= num_samples"
        assert cfg.planner.num_elites > 1, "num_elites should be > 1"

    # -------------------------
    # Build Logger, Agent, and start the evaluation loop
    logger = Logger(cfg)

    agent = GC_Agent(cfg, model, dset=dset, preprocessor=preprocessor)
    cfg.task_indices, cfg.episodes_per_task = compute_task_distribution(cfg)
    log.info(f"Rank {cfg.rank}: \n {cfg.task_indices=} \n {cfg.episodes_per_task=}")
    # The multitask wrapper allows to iterate over the task-specific envs
    env = make_env(cfg)
    evaluator = PlanEvaluator(cfg, agent)
    results = dict()
    processed_episodes = set()
    for task_pos, (task_idx, episodes) in enumerate(zip(cfg.task_indices, cfg.episodes_per_task)):
        (
            ep_rewards,
            ep_successes,
            ep_expert_successes,
            ep_success_dists,
            ep_end_distances,
            ep_end_distances_xyz,
            ep_end_distances_orientation,
            ep_end_distances_closure,
            ep_times,
            ep_state_distances,
            ep_total_lpips,
            ep_total_emb_l2,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        # We want each episode to be independent from the others, even of the same task
        for i, ep in enumerate(episodes):
            log.info(
                f"🎯 Evaluating task {cfg.tasks[task_idx]}, episode ({i + 1}/{len(cfg.episodes_per_task[task_pos])})"
            )
            episode_start_time = time()
            (
                expert_success,
                success,
                ep_reward,
                success_dist,
                end_distance,
                end_distance_xyz,
                end_distance_orientation,
                end_distance_closure,
                state_dist,
                total_lpips,
                total_emb_l2,
            ) = evaluator.eval(cfg, agent, env, task_idx=task_idx, ep=ep)
            dist.barrier(group=process_group)  # should not provoke any error
            episode_end_time = time()
            # Check for duplicate task and episode index
            if (task_idx, ep) in processed_episodes:
                continue  # Skip duplicate dummy episodes logging
            processed_episodes.add((task_idx, ep))
            ep_rewards.append(ep_reward)
            ep_successes.append(success)
            ep_expert_successes.append(expert_success)
            ep_success_dists.append(success_dist)
            ep_end_distances.append(end_distance)
            ep_end_distances_xyz.append(end_distance_xyz)
            ep_end_distances_orientation.append(end_distance_orientation)
            ep_end_distances_closure.append(end_distance_closure)
            ep_times.append(episode_end_time - episode_start_time)
            ep_state_distances.append(state_dist)
            ep_total_lpips.append(total_lpips)
            ep_total_emb_l2.append(total_emb_l2)
        # Mean over episodes for each task
        results.update(
            {
                f"episode_reward+{cfg.tasks[task_idx]}": (np.nansum(ep_rewards), len(ep_rewards)),
                f"episode_success+{cfg.tasks[task_idx]}": (np.nansum(ep_successes), len(ep_successes)),
                f"ep_expert_succ+{cfg.tasks[task_idx]}": (np.nansum(ep_expert_successes), len(ep_expert_successes)),
                f"ep_succ_dist+{cfg.tasks[task_idx]}": (np.nansum(ep_success_dists), len(ep_success_dists)),
                f"ep_end_dist+{cfg.tasks[task_idx]}": (np.nansum(ep_end_distances), len(ep_end_distances)),
                f"ep_end_dist_xyz+{cfg.tasks[task_idx]}": (np.nansum(ep_end_distances_xyz), len(ep_end_distances_xyz)),
                f"ep_end_dist_orientation+{cfg.tasks[task_idx]}": (
                    np.nansum(ep_end_distances_orientation),
                    len(ep_end_distances_orientation),
                ),
                f"ep_end_dist_closure+{cfg.tasks[task_idx]}": (
                    np.nansum(ep_end_distances_closure),
                    len(ep_end_distances_closure),
                ),
                f"ep_time+{cfg.tasks[task_idx]}": (np.nansum(ep_times), len(ep_times)),
                f"ep_state_dist+{cfg.tasks[task_idx]}": (np.nansum(ep_state_distances), len(ep_state_distances)),
                f"ep_total_lpips+{cfg.tasks[task_idx]}": (np.nansum(ep_total_lpips), len(ep_total_lpips)),
                f"ep_total_emb_l2+{cfg.tasks[task_idx]}": (np.nansum(ep_total_emb_l2), len(ep_total_emb_l2)),
            }
        )

    if cfg.distributed.distribute_multitask_eval:
        all_results = [None] * cfg.world_size
        log.info(f"{rank=}: {results=}")
        if rank == 0:
            dist.gather_object(results, object_gather_list=all_results, dst=0, group=process_group)
        else:
            dist.gather_object(results, object_gather_list=None, dst=0, group=process_group)
            combined_results = {}
        if rank == 0:
            combined_results = aggregate_results(cfg, all_results)
            observed_episode_counts = _episode_counts(all_results, cfg.tasks)
            log.info(f"{combined_results=}")
    else:
        combined_results = {key: value[0] / value[1] if value[1] > 0 else 0 for key, value in results.items()}
        observed_episode_counts = _episode_counts([results], cfg.tasks)
    if cfg.rank == 0:
        metrics = {"total_time": time() - start_time}
        # Create average over tasks
        logger.pprint_multitask(combined_results | metrics, cfg)
        reported_metrics = logger.log(combined_results | metrics, multitask=cfg.task_specification.multitask)
        if planning_provenance is not None:
            planning_result = build_complete_planning_result(
                planning_provenance,
                metrics=reported_metrics,
                observed_episode_counts=observed_episode_counts,
            )
            result_path = write_complete_planning_result(planning_result)
            log.info(
                "📌 Published complete planning result for checkpoint %s to %s",
                planning_provenance["checkpoint_id"],
                result_path,
            )
        return reported_metrics


def init_module(
    folder,
    checkpoint,
    module_name,
    model_kwargs,
    device,
    cfgs_data=None,
    wrapper_kwargs=None,
    action_dim=None,
    proprio_dim=None,
    preprocessor=None,
):
    """
    Build (frozen) model and initialize from pretrained checkpoint
    """
    model = importlib.import_module(f"{module_name}").init_module(
        folder=folder,
        checkpoint=checkpoint,
        model_kwargs=model_kwargs,
        device=device,
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        preprocessor=preprocessor,
        cfgs_data=cfgs_data,
        wrapper_kwargs=wrapper_kwargs,
    )
    return model
