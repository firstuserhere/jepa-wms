# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import hashlib
import json
import os
import random
import re
import tempfile

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass
import time
from collections import defaultdict

import imageio
import lpips as lpips_lib
import matplotlib
import numpy as np
import submitit
import torch
import torch.multiprocessing as mp
import wandb
from einops import rearrange
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from app.plan_common.datasets.preprocessor import Preprocessor
from app.plan_common.datasets.transforms import make_inverse_transforms, make_transforms
from app.plan_common.datasets.utils import init_data
from app.plan_common.models.wm_heads import (
    WorldModelPoseReadoutHead,
    WorldModelRewardReadoutHead,
    WorldModelViTImageHead,
)
from app.vjepa_wm.utils import (
    build_plan_eval_args,
    build_unroll_decode_eval_args,
    clean_state_dict,
    fetch_checkpoint,
    init_opt,
    init_video_model,
    load_checkpoint,
)
from app.vjepa_wm.video_wm import VideoWM
from evals.main_distributed import launch_evals_with_parsed_args as launch_evals
from src.datasets.utils.utils import get_dataset_paths
from src.utils.cluster import slurm_account_partition_and_qos
from src.utils.checkpointing import (
    CheckpointManager,
    ResumeContractError,
    atomic_json_dump,
    atomic_torch_save,
    build_checkpoint_v2,
    capture_rng_state,
    create_checkpoint_manifest,
    gather_rng_states,
    is_v2_checkpoint,
    restore_rng_state,
    restore_rank_rng_state,
    sha256_file,
    training_state_from_checkpoint,
    validate_checkpoint_v2,
    validate_resume_contract,
)
from src.utils.distributed import init_distributed
from src.utils.dataset_manifest import (
    verify_auxiliary_runtime_binding,
    verify_droid_runtime_binding,
)
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer
from src.utils.mfu import training_performance_stats
from src.utils.training_telemetry import (
    EffectiveMFUAccumulator,
    MFU_FORMULA_ID,
    MFU_FORMULA_VERSION,
    PantheonWandbHeartbeat,
    telemetry_provenance,
    telemetry_snapshot_document,
)
from src.utils.planning_promotion import (
    drain_planning_evaluations,
    load_planning_evaluation_registry,
    mark_planning_evaluations_launched,
    poll_planning_evaluations,
    register_planning_evaluations,
)
from src.utils.yaml_utils import convert_to_dict_recursive, dump_yaml, expand_env_vars
from src.utils.wandb_utils import generate_wandb_run_id

# --
log_timings = True
log_freq = 10
checkpoint_freq = 1
DEFAULT_EVAL_FREQ = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__)


def _run_rank0_planning_controller(rank, operation, description):
    """Run one checkpoint-controller mutation on rank zero and fan out errors."""

    report = None
    error = [None]
    if rank == 0:
        try:
            report = operation()
        except Exception as controller_error:
            error[0] = f"{type(controller_error).__name__}: {controller_error}"
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(f"{description} failed: {error[0]}")
    return report


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #
    args = expand_env_vars(args)
    folder = args.get("folder")
    checkpoint_folder = args.get("checkpoint_folder", folder)
    os.makedirs(checkpoint_folder, exist_ok=True)
    cfgs_checkpointing = args.get("checkpointing", {})
    checkpointing_enabled = cfgs_checkpointing.get("enabled", False)
    strict_continuation = cfgs_checkpointing.get("strict_continuation", False)
    checkpoint_load_mode = cfgs_checkpointing.get("load_mode", "resume")
    if checkpoint_load_mode not in ("resume", "fork"):
        raise ValueError("checkpointing.load_mode must be 'resume' or 'fork'")
    checkpoint_save_every = int(cfgs_checkpointing.get("save_every_epochs", checkpoint_freq))
    if checkpoint_save_every <= 0:
        raise ValueError("checkpointing.save_every_epochs must be positive")
    checkpoint_keep_recent = int(cfgs_checkpointing.get("keep_recent", 0))
    if checkpoint_keep_recent < 0:
        raise ValueError("checkpointing.keep_recent must be non-negative")
    checkpoint_epoch_boundary_only = bool(cfgs_checkpointing.get("epoch_boundary_only", True))
    checkpoint_loss_budget_seconds = float(cfgs_checkpointing.get("max_epoch_boundary_seconds", 300.0))
    if not checkpoint_epoch_boundary_only:
        raise ValueError("JEPA-WM continuation checkpoints are supported only at completed epoch boundaries")
    if checkpoint_loss_budget_seconds <= 0:
        raise ValueError("checkpointing.max_epoch_boundary_seconds must be positive")
    if checkpointing_enabled and checkpoint_save_every != 1:
        raise ValueError(
            "Pantheon-safe epoch-boundary recovery requires checkpointing.save_every_epochs=1"
        )
    # -- META
    cfgs_meta = args.get("meta")
    load_model = cfgs_meta.get("load_checkpoint") or resume_preempt
    load_opt_scale_epoch = cfgs_meta.get("load_opt_scale_epoch", False)
    freeze_encoder = cfgs_meta.get("freeze_encoder", True)
    r_file = cfgs_meta.get("read_checkpoint", None)
    pretrained_path = cfgs_meta.get("pretrained_path", None)
    strict_checkpoint_weights = cfgs_meta.get("strict_checkpoint_weights", False)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)

    eval_freq = cfgs_meta.get("eval_freq", DEFAULT_EVAL_FREQ)
    plan_only_eval_mode = cfgs_meta.get("plan_only_eval_mode", False)
    unroll_decode_eval_only_mode = cfgs_meta.get("unroll_decode_eval_only_mode", False)
    light_eval_only_mode = cfgs_meta.get("light_eval_only_mode", False)
    rollout_only_eval_mode = cfgs_meta.get("rollout_only_eval_mode", False)

    # -- LIGHT EVALS (keep as subconfigs, extract only frequently-checked flags)
    cfgs_data_traj_rollout_eval = cfgs_meta.get("data_traj_rollout_eval", {})
    cfgs_energy_landscape_eval = cfgs_meta.get("energy_landscape_eval", {})
    do_data_traj_rollout_eval = cfgs_data_traj_rollout_eval.get("do_data_traj_rollout_eval", False)
    do_energy_landscape_eval = cfgs_energy_landscape_eval.get("do_energy_landscape_eval", False)
    data_traj_decode_gt = cfgs_data_traj_rollout_eval.get("data_traj_decode_gt", False)

    light_eval_freq = cfgs_meta.get("light_eval_freq", 100)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    quick_debug = cfgs_meta.get("quick_debug", False)
    which_dtype = cfgs_meta.get("dtype")
    logger.info(f"⚙️  Using dtype: {which_dtype}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- EVALS
    cfgs_plan_evals = args.get("evals", None)
    cfgs_unroll_decode_evals = args.get("unroll_decode_evals", None)

    # -- MODEL (extract only fields needed outside init_video_model)
    cfgs_model = args.get("model")

    # Extract heads config
    cfgs_heads = cfgs_model.get("heads_cfg", {})
    heads_architectures = cfgs_heads.get("architectures", {})
    pretrain_dec_path = cfgs_heads.get("pretrain_dec_path", None)
    new_path_heads = cfgs_heads.get("new_path_heads", {})

    # Fields needed for training loop
    rollout_cfg = cfgs_model.get("rollout_cfg", {})
    rollout_steps = rollout_cfg.get("rollout_steps", 0)
    train_rollout_prefixes = rollout_cfg.get("train_rollout_prefixes", "random")
    rollout_stop_gradient = rollout_cfg.get("rollout_stop_gradient", True)
    ctxt_window_train_rollout = rollout_cfg.get("ctxt_window_train_rollout", 8)
    do_parallel_rollout = rollout_cfg.get("do_parallel_rollout", False)
    do_sequential_rollout = rollout_cfg.get("do_sequential_rollout", True)
    prepend_gt_rollout_parallel = rollout_cfg.get("prepend_gt", False)

    sampling_scheduler_cfg = rollout_cfg.get("sampling_scheduler", {})
    sampling_scheduler_type = sampling_scheduler_cfg.get("type", "linear")
    sampling_scheduler_start = sampling_scheduler_cfg.get("start", 0.0)
    sampling_scheduler_end = sampling_scheduler_cfg.get("end", 0.0)

    # Derived values needed for dimensions (computed values used across the code)
    action_tokens = cfgs_model.get("action_encoder", {}).get("action_tokens", 1)
    proprio_tokens = cfgs_model.get("proprio_encoder", {}).get("proprio_tokens", 1)
    action_emb_dim = cfgs_model.get("action_encoder", {}).get("action_emb_dim", 0)
    proprio_emb_dim = cfgs_model.get("proprio_encoder", {}).get("proprio_emb_dim", 0)
    use_proprio = proprio_tokens > 0 or proprio_emb_dim > 0
    use_action = action_tokens > 0 or action_emb_dim > 0
    tubelet_size_enc = cfgs_model.get("tubelet_size_enc", 2)

    cfgs_wm_encoding = cfgs_model.get("wm_encoding", {})

    if cfgs_wm_encoding.get("dup_image", False):
        assert tubelet_size_enc == 1, "Batchify video only works with tubelet_size_enc=1"

    # -- DATA (extract only fields needed outside init_data)
    cfgs_data = args.get("data")
    cfgs_validation = cfgs_data.get("validation", {})
    cfgs_loader = cfgs_data.get("loader", {})
    if strict_continuation and cfgs_loader.get("persistent_workers", False):
        logger.warning(
            "Strict continuation recreates DataLoader workers at every epoch boundary; "
            "forcing data.loader.persistent_workers=false."
        )
        cfgs_loader["persistent_workers"] = False
    cfgs_custom = cfgs_data.get("custom", {})
    cfgs_droid = cfgs_data.get("droid", {})

    # Compute dataset paths
    datasets = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights", None)
    if datasets_weights is not None:
        assert len(datasets_weights) == len(datasets), "Must have one sampling weight specified for each dataset"

    dataset_type = cfgs_data.get("dataset_type", "custom")
    if dataset_type.lower() == "mixed_dataset":
        dataset_paths = datasets
    else:
        dataset_paths = get_dataset_paths(datasets)

    dataset_manifest = None
    dataset_manifest_path = cfgs_checkpointing.get("dataset_manifest_path")
    if dataset_manifest_path:
        if not os.path.isfile(dataset_manifest_path):
            raise FileNotFoundError(f"Dataset manifest not found: {dataset_manifest_path}")
        with open(dataset_manifest_path, "r", encoding="utf-8") as stream:
            dataset_manifest = json.load(stream)
        verification = dataset_manifest.get("verification", {})
        if verification and not verification.get("complete", False):
            raise RuntimeError(f"Dataset manifest is incomplete: {dataset_manifest_path}")
        if cfgs_checkpointing.get("require_dataset_manifest", False) and not verification.get(
            "files_checked", False
        ):
            raise RuntimeError(f"Dataset manifest was not file-verified: {dataset_manifest_path}")
        if cfgs_checkpointing.get("require_source_inventory", False) and not verification.get(
            "source_listing_matches_staged", False
        ):
            raise RuntimeError(
                f"Dataset manifest does not match a verified official source inventory: {dataset_manifest_path}"
            )
        if not dataset_manifest.get("dataset_fingerprint"):
            raise RuntimeError(f"Dataset manifest has no dataset_fingerprint: {dataset_manifest_path}")
    elif cfgs_checkpointing.get("require_dataset_manifest", False):
        raise RuntimeError("checkpointing.require_dataset_manifest=true requires dataset_manifest_path")
    elif checkpointing_enabled:
        dataset_manifest = {
            "dataset": datasets,
            "paths": dataset_paths,
            "verified": False,
            "warning": "No external dataset manifest was supplied",
        }

    val_datasets = cfgs_validation.get("val_datasets", [])
    val_dataset_paths = get_dataset_paths(val_datasets) if val_datasets else None

    # val_datasets_1 subconfig (for second validation set)
    val_datasets_1 = cfgs_validation.get("val_datasets_1", None)
    val_datasets_1_paths = None
    if val_datasets_1 is not None:
        val_datasets_1_paths = get_dataset_paths(val_datasets_1.get("names"))

    if dataset_manifest is not None and cfgs_checkpointing.get(
        "require_runtime_dataset_binding", cfgs_checkpointing.get("require_dataset_manifest", False)
    ):
        if datasets == ["DROID"]:
            dataset_manifest["runtime_binding"] = verify_droid_runtime_binding(
                dataset_manifest, dataset_paths
            )
        if "Franka_hf" in val_datasets:
            franka_index = val_datasets.index("Franka_hf")
            dataset_manifest.setdefault("runtime_auxiliary_bindings", {})["Franka_hf"] = (
                verify_auxiliary_runtime_binding(
                    dataset_manifest,
                    "Franka_hf",
                    val_dataset_paths[franka_index],
                )
            )

    # Fields used outside init_data
    frameskip = cfgs_custom.get("frameskip", True)
    action_skip = cfgs_custom.get("action_skip", 1)
    state_skip = cfgs_custom.get("state_skip", 1)
    img_size = cfgs_data.get("img_size", 256)
    num_workers = cfgs_loader.get("num_workers", 1)
    filter_first_episodes = cfgs_custom.get("filter_first_episodes", None)
    val_viz_rank0_loader = cfgs_validation.get("val_viz_rank0_loader", False)

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")

    # -- LOSS
    cfgs_loss = args.get("loss")

    # -- OPTIMIZATION (simplify main_optimizer logic)
    cfgs_opt = args.get("optimization")
    train_heads = cfgs_opt["train_heads"]

    main_optimizer = cfgs_opt["main_optimizer"]
    if main_optimizer == "transition_model":
        num_epochs = cfgs_opt["transition_model"]["num_epochs"]
        ipe = cfgs_opt["transition_model"]["iterations_per_epoch"]
        train_predictor = True
        train_heads_on_predictor = False
    else:  # image_head | state_head | reward_head
        num_epochs = cfgs_opt["heads"][main_optimizer]["num_epochs"]
        ipe = cfgs_opt["heads"][main_optimizer]["iterations_per_epoch"]
        train_predictor = cfgs_opt["heads"]["train_predictor"]
        train_heads_on_predictor = cfgs_opt["heads"]["train_heads_on_predictor"]

    if rollout_only_eval_mode:
        if not light_eval_only_mode:
            raise ValueError("meta.rollout_only_eval_mode requires meta.light_eval_only_mode=true")
        if plan_only_eval_mode or unroll_decode_eval_only_mode:
            raise ValueError("rollout-only qualification cannot be combined with another eval-only mode")
        if not checkpointing_enabled:
            raise ValueError("rollout-only qualification requires checkpointing.enabled=true")
        if not cfgs_checkpointing.get("rollout_promotion", {}).get("enabled", False):
            raise ValueError("rollout-only qualification requires checkpointing.rollout_promotion.enabled=true")
        if num_epochs != 1 or ipe != 1:
            raise ValueError(
                "rollout-only qualification must set transition_model num_epochs=1 and "
                "iterations_per_epoch=1"
            )

    # -- LOGGING
    cfgs_logging = args.get("logging")
    tag = cfgs_logging.get("write_tag", "jepa")
    latest_format = cfgs_logging.get("latest_format", "pth.tar")
    cfgs_wandb = cfgs_logging.get("wandb")
    cfgs_mfu = cfgs_logging.get("mfu", {})
    mfu_enabled = bool(cfgs_mfu.get("enabled", False))
    mfu_peak_dense_tflops = float(cfgs_mfu.get("peak_dense_tflops", 0.0))
    mfu_measure_epoch = int(cfgs_mfu.get("measure_epoch", 0))
    mfu_measure_iteration = int(cfgs_mfu.get("measure_iteration", 2))
    mfu_telemetry_interval_steps = int(cfgs_mfu.get("telemetry_interval_steps", 1))
    mfu_heartbeat_interval_seconds = float(cfgs_mfu.get("heartbeat_interval_seconds", 10.0))
    if mfu_enabled:
        if mfu_peak_dense_tflops <= 0:
            raise ValueError("logging.mfu.peak_dense_tflops must be positive when MFU is enabled")
        if mfu_measure_epoch < 0 or mfu_measure_iteration < 0:
            raise ValueError("logging.mfu measurement coordinates must be non-negative")
        if mfu_telemetry_interval_steps <= 0 or mfu_heartbeat_interval_seconds <= 0:
            raise ValueError("MFU telemetry and heartbeat intervals must be positive")

    if light_eval_only_mode:
        light_eval_freq = 1
    if plan_only_eval_mode or light_eval_only_mode or quick_debug or unroll_decode_eval_only_mode:
        # Qualification must retain the configured dataset rather than silently
        # reducing it to the ten-episode interactive/debug subset.
        if not rollout_only_eval_mode:
            filter_first_episodes = 10
        num_workers = 0

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass
    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"🚀 Initialized (rank/world-size) {rank}/{world_size}")

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        # Submitit pins each process to one visible device.  torchrun exposes
        # all local devices and communicates the process binding via LOCAL_RANK.
        local_rank = 0 if "SLURM_LOCALID" in os.environ else int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    train_log_file = os.path.join(folder, f"log_r{rank}.csv")
    pref_tag = f"{tag}-" if tag else ""
    latest_file = pref_tag + f"latest.{latest_format}"
    latest_path = os.path.join(checkpoint_folder, latest_file)
    checkpoint_manager = (
        CheckpointManager(
            checkpoint_folder,
            prefix=tag or "jepa",
            keep_recent=checkpoint_keep_recent,
        )
        if checkpointing_enabled
        else None
    )
    managed_latest_path = None
    if checkpoint_manager is not None:
        try:
            managed_latest_path = os.fspath(checkpoint_manager.resolve("latest", verify=True))
        except FileNotFoundError:
            pass
    finetune = pretrained_path is not None
    logger.info(f"{'🔧 Finetuning mode' if finetune else '🆕 Training from scratch'}")
    # finetune covers the case of training heads on top of frozen transition_model
    # if cfgs_model["pretrained_path"] is None, i.e. training head just on encoder
    head_training_mode = main_optimizer in ["image_head", "state_head"]
    resume_path = managed_latest_path or (latest_path if os.path.exists(latest_path) else None)
    resume = resume_path is not None
    resume_finetune = resume and finetune
    resume_latest = resume and not finetune
    if resume_finetune:
        logger.info("♻️  Resuming from checkpoint")
    load_path = None
    if load_model:
        if resume_finetune:
            load_path = os.path.join(folder, r_file) if r_file is not None else resume_path
            load_opt_scale_epoch = not head_training_mode
        elif resume_latest:
            load_path = resume_path
            load_opt_scale_epoch = not head_training_mode
        else:  # not resuming, i.e. not os.path.exists(latest_path)
            load_path = pretrained_path
        if load_path is None or not os.path.exists(load_path):
            load_path = None
            load_model = False

    train_csv_logger, eval_csv_logger = None, None

    def create_csv_logger(losses, total_stats, train=False):
        if train:
            csv_log_file = train_log_file
        else:
            if light_eval_only_mode:
                csv_log_file = os.path.join(folder, f"light_eval_only_eval_r{rank}.csv")
            else:
                csv_log_file = os.path.join(folder, f"eval_r{rank}.csv")
        # Get all unique keys from eval_losses and eval_total_stats
        excluded_keys = [
            "eval_data/image_rollouts",
            "eval_data/image_rollouts_noisy_actions",
            "eval_data/image_animated_rollout",
        ]
        all_keys = {key for key in list(losses.keys()) + list(total_stats.keys()) if key not in excluded_keys}
        # Sort keys lexicographically for consistent order
        sorted_keys = sorted(all_keys)
        # Create a format tuple for each key
        new_columns = [("%.5f", key) for key in sorted_keys]
        # Initialize the logger with new columns
        if train:
            global train_csv_logger_columns
            train_csv_logger_columns = ["epoch", "itr", "loss", "gpu-time(ms)", "iter-time(ms)"] + sorted_keys
        else:
            global eval_csv_logger_columns
            eval_csv_logger_columns = ["epoch", "itr"] + sorted_keys
        return CSVLogger(csv_log_file, ("%d", "epoch"), ("%d", "itr"), *new_columns)

    # -- init data-loaders/samplers
    transform = make_transforms(
        img_size=img_size,
        **cfgs_data_aug,
    )
    inverse_transform = make_inverse_transforms(img_size=img_size, **cfgs_data_aug)

    # Prepare data kwargs from config, flattening nested structures and filtering out non-init_data fields
    excluded_keys = [
        "datasets",
        "val_datasets",
        "img_size",
        # subfields of cfgs_data
        "validation",
        "loader",
        "custom",
        "droid",
    ]
    data_kwargs = {k: v for k, v in cfgs_data.items() if k not in excluded_keys}
    # Add fields from nested structures
    data_kwargs.update(cfgs_validation)
    data_kwargs.update(cfgs_loader)
    data_kwargs.update(cfgs_custom)
    data_kwargs.update(cfgs_droid)
    # Add computed/override parameters
    data_kwargs.update(
        {
            "data_paths": dataset_paths,
            "val_data_paths": val_dataset_paths,
            "transform": transform,
            "world_size": world_size,
            "rank": rank,
            # overridden by quick_debug
            "filter_first_episodes": filter_first_episodes,  # Potentially overridden by mode
            "num_workers": num_workers,
        }
    )

    val_data_iters = []
    (
        dataset,
        val_dataset,
        traj_dataset,
        val_traj_dataset,
        unsupervised_loader,
        val_unsupervised_loader,
        unsupervised_sampler,
        viz_val_data_loader,
    ) = init_data(**data_kwargs)
    val_data_iters.append((val_dataset, val_traj_dataset, val_unsupervised_loader))

    if val_datasets_1:
        # Reuse data_kwargs and override val_datasets_1-specific fields
        data_kwargs_1 = data_kwargs.copy()
        data_kwargs_1.update(
            {
                "dset_fraction": 1,
                "val_dset_fraction": 1,
                "data_paths": val_datasets_1_paths,
                "val_data_paths": val_datasets_1_paths,
                "batch_size": val_datasets_1.get("batch_size", 4),
                "drop_last": val_datasets_1.get("drop_last", True),
                "fps": val_datasets_1.get("fps", 4),
                "dataset_fpcs": val_datasets_1.get("fpcs", [8]),
                "val_dataset_fpcs": val_datasets_1.get("fpcs", [8]),
                "camera_views": val_datasets_1.get("camera_views", ["exterior_image_2_left"]),
                "droid_to_rcasa_action_format": val_datasets_1.get("droid_to_rcasa_action_format", 1),
            }
        )
        _, val_dataset_1, _, val_traj_dataset_1, _, val_unsupervised_loader_1, _, viz_val_data_loader = init_data(
            **data_kwargs_1
        )
        val_data_iters.append((val_dataset_1, val_traj_dataset_1, val_unsupervised_loader_1))
    if dataset_type == "custom" and traj_dataset is not None:
        preprocessor = Preprocessor(
            action_mean=traj_dataset.action_mean,
            action_std=traj_dataset.action_std,
            state_mean=traj_dataset.state_mean,
            state_std=traj_dataset.state_std,
            proprio_mean=traj_dataset.proprio_mean,
            proprio_std=traj_dataset.proprio_std,
            transform=transform,
            inverse_transform=inverse_transform,
        )
    else:
        preprocessor = None

    _dlen = len(unsupervised_loader)
    if ipe is None:
        ipe = _dlen
    logger.info(f"📊 Iterations per epoch: {ipe} (dataset size: {_dlen})")
    if main_optimizer == "transition_model":
        cfgs_opt["transition_model"]["iterations_per_epoch"] = ipe
    elif main_optimizer == "image_head":
        cfgs_opt["heads"]["image_head"]["iterations_per_epoch"] = ipe
    elif main_optimizer == "state_head":
        cfgs_opt["heads"]["state_head"]["iterations_per_epoch"] = ipe

    # Logger
    class Trainer:
        def __init__(self, config):
            config = config or {}
            if quick_debug:
                config["debug"] = quick_debug
            self.config = config
            self.ipe = ipe
            self.wandb_required = config.get("required", False)
            self.wandb_required_online = config.get("required_online", False)
            self.use_wandb = config.get("use_wandb", False) or self.wandb_required
            self.disable_wandb_media = config.get("disable_wandb_media", False)
            self.log_media_locally = config.get("log_media_locally", False)
            self.wandb_run_id = None
            self.local_log_dir = None
            if self.log_media_locally and rank == 0:
                self.local_log_dir = os.path.join(folder, "local_logs")
                os.makedirs(self.local_log_dir, exist_ok=True)
            if self.use_wandb and rank == 0:
                wandb_mode = os.environ.get("WANDB_MODE", "online").lower()
                if self.wandb_required_online and wandb_mode != "online":
                    raise RuntimeError(
                        f"Online W&B is required for this run, but WANDB_MODE={wandb_mode!r}"
                    )
                if self.wandb_required and not os.environ.get("WANDB_API_KEY") and os.environ.get(
                    "WANDB_MODE", ""
                ).lower() != "offline":
                    raise RuntimeError("W&B is required, but WANDB_API_KEY is not present in the environment")
                configured_project = (
                    config.get("project", "vjepa_wm")
                    if not config.get("debug", False)
                    else config.get("debug_project", "vjepa_wm_debug")
                )
                project_name = os.environ.get("WANDB_PROJECT", "").strip() or configured_project
                run_metadata_dir = os.path.join(checkpoint_folder, "run_metadata")
                os.makedirs(run_metadata_dir, exist_ok=True)
                wandb_run_id_file = os.path.join(run_metadata_dir, "wandb_run_id.txt")
                expected_wandb_run_id = os.environ.get("WANDB_RUN_ID", "").strip() or None
                requested_resume_mode = os.environ.get("WANDB_RESUME", "allow").strip().lower()
                if self.wandb_required and requested_resume_mode != "allow":
                    raise RuntimeError(
                        "Pantheon training requires WANDB_RESUME=allow for stable recovery continuity"
                    )
                if os.path.exists(wandb_run_id_file):
                    with open(wandb_run_id_file, "r") as f:
                        wandb_run_id = f.read().strip()
                    if not wandb_run_id:
                        raise RuntimeError(f"Empty W&B run ID file: {wandb_run_id_file}")
                    if expected_wandb_run_id and wandb_run_id != expected_wandb_run_id:
                        raise RuntimeError(
                            f"W&B run ID sidecar {wandb_run_id!r} does not match "
                            f"WANDB_RUN_ID={expected_wandb_run_id!r}"
                        )
                else:
                    latest_alias = (
                        checkpoint_manager.read_alias("latest", verify=False)
                        if checkpoint_manager is not None
                        else None
                    )
                    alias_wandb_id = (
                        latest_alias.get("checkpoint", {}).get("metadata", {}).get("wandb_run_id")
                        if latest_alias is not None
                        else None
                    )
                    if alias_wandb_id:
                        wandb_run_id = str(alias_wandb_id)
                        if expected_wandb_run_id and wandb_run_id != expected_wandb_run_id:
                            raise RuntimeError(
                                f"Checkpoint W&B identity {wandb_run_id!r} does not match "
                                f"WANDB_RUN_ID={expected_wandb_run_id!r}"
                            )
                    elif latest_alias is not None:
                        raise RuntimeError(
                            "A latest checkpoint exists but its W&B identity sidecar is missing; "
                            "refusing to split an exact continuation across runs"
                        )
                    else:
                        wandb_run_id = expected_wandb_run_id or generate_wandb_run_id()
                wandb.init(
                    project=project_name,
                    entity=config.get("entity"),
                    group=config.get("group"),
                    tags=config.get("tags"),
                    id=wandb_run_id,
                    resume="allow",
                    dir=folder,
                    config=convert_to_dict_recursive(args),
                )
                self.wandb_run_id = wandb_run_id
                if os.path.exists(wandb_run_id_file):
                    logger.info(f"Resuming Wandb run {wandb_run_id}")
                else:
                    descriptor, temporary_path = tempfile.mkstemp(
                        prefix=".wandb_run_id.", suffix=".tmp", dir=run_metadata_dir
                    )
                    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                        stream.write(wandb_run_id)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary_path, wandb_run_id_file)
                if self.wandb_required_online and not getattr(wandb.run, "url", None):
                    raise RuntimeError("W&B online initialization returned no canonical run URL")
                wandb.run.name = os.path.basename(folder)
                wandb.define_metric("train/global_step")
                wandb.define_metric("*", step_metric="train/global_step")
                self.job_set = set()

        def log(
            self,
            epoch,
            itr,
            losses,
            total_stats,
            eval_losses=None,
            eval_total_stats=None,
            image_stats=None,
            global_step=None,
        ):
            log_dict = {
                "epoch": epoch + 1,
                "itr": itr,
                "train/global_step": epoch * self.ipe + itr + 1 if global_step is None else global_step,
            }
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().item()
                log_dict[key] = value
            if "loss" in log_dict:
                log_dict["train/loss"] = float(log_dict["loss"])
            for key, value in total_stats.items():
                log_dict[key] = value
            if eval_losses is not None:
                for key, value in eval_losses.items():
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().item()
                    log_dict[key] = value
                validation_loss_keys = sorted(
                    key for key in eval_losses if key == "loss" or key.endswith("/loss")
                )
                if validation_loss_keys:
                    value = eval_losses[validation_loss_keys[0]]
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().item()
                    log_dict["val/loss"] = float(value)
            if eval_total_stats is not None:
                for key, value in eval_total_stats.items():
                    log_dict[key] = value
            if image_stats:  # not None or nonempty
                if self.log_media_locally and rank == 0:
                    self.log_media_local(image_stats, epoch, itr)
                if not self.disable_wandb_media:
                    log_dict.update(image_stats)
            if "loss" in log_dict.keys() and itr % log_freq == 0:
                logger.info("[%d, %5d] " "loss: %.3f | " % (epoch + 1, itr, log_dict["loss"]))
            if self.use_wandb and rank == 0:
                wandb.log(log_dict, step=log_dict["train/global_step"])
                if "perf/mfu_dense" in log_dict:
                    current_mfu = float(log_dict["perf/mfu_dense"])
                    wandb.run.summary["perf/mfu_dense_last"] = current_mfu
                    previous_max = wandb.run.summary.get("perf/mfu_dense_max")
                    wandb.run.summary["perf/mfu_dense_max"] = (
                        current_mfu if previous_max is None else max(float(previous_max), current_mfu)
                    )

        def log_media_local(self, image_stats, epoch=None, itr=None):
            step = epoch * ipe + itr
            for key, value in image_stats.items():
                subfolder = os.path.join(self.local_log_dir, "/".join(key.split("/")))
                os.makedirs(subfolder, exist_ok=True)
                filename = f"{step}.gif" if isinstance(value, wandb.Video) else f"{step}.pdf"
                if isinstance(value, wandb.Video):
                    frames = value._prepare_video(value.data)
                    duration = 1.0 / 10  # assuming 10 FPS
                    imageio.mimsave(os.path.join(subfolder, filename), frames, duration=duration, loop=0)
                elif isinstance(value, wandb.Image):
                    value.image.save(os.path.join(subfolder, filename))
                elif isinstance(value, matplotlib.figure.Figure):
                    value.savefig(os.path.join(subfolder, filename), bbox_inches=None)

    trainer = Trainer(cfgs_wandb)
    if checkpointing_enabled and not trainer.use_wandb:
        raise RuntimeError("Schema-v2 research checkpoints require logging.wandb.required=true")
    if checkpoint_manager is not None:
        repair_report = _run_rank0_planning_controller(
            rank,
            checkpoint_manager.repair_embedded_promotions,
            "checkpoint role repair",
        )
        if rank == 0 and trainer.use_wandb:
            repaired_rollout = repair_report.get("repaired", {}).get("best_rollout")
            if repaired_rollout is not None:
                best_rollout_alias = checkpoint_manager.read_alias("best_rollout", verify=True)
                wandb.run.summary["best_rollout/checkpoint_id"] = best_rollout_alias["checkpoint"][
                    "object_id"
                ]
                wandb.run.summary["best_rollout/value"] = best_rollout_alias["promotion"][
                    "metric_value"
                ]

    # -- init model
    if use_action:
        actions_per_vid_feat = tubelet_size_enc * frameskip // action_skip
        model_action_dim = traj_dataset.action_dim * tubelet_size_enc * frameskip // action_skip
    else:
        actions_per_vid_feat, model_action_dim = None, None
    if use_proprio:
        model_proprio_dim = traj_dataset.proprio_dim * tubelet_size_enc // state_skip
    else:
        model_proprio_dim = None

    # Prepare model kwargs by flattening nested configs and filtering out non-init_video_model fields
    excluded_keys = [
        "rollout_cfg",
        "heads_cfg",
        "pretrained_path",
        "visual_encoder",
        "action_encoder",
        "proprio_encoder",
        "predictor",
        "wm_encoding",
        "attn",
    ]
    model_kwargs = {k: v for k, v in cfgs_model.items() if k not in excluded_keys}
    if "visual_encoder" in cfgs_model:
        model_kwargs.update(cfgs_model["visual_encoder"])
    if "action_encoder" in cfgs_model:
        model_kwargs.update(cfgs_model["action_encoder"])
    if "proprio_encoder" in cfgs_model:
        model_kwargs.update(cfgs_model["proprio_encoder"])
    if "predictor" in cfgs_model:
        model_kwargs.update(cfgs_model["predictor"])
    model_kwargs.update(
        {
            "device": device,
            "img_size": img_size,
            "action_dim": model_action_dim,  # Computed from dataset
            "proprio_dim": model_proprio_dim,  # Computed from dataset
            "cfgs_attn_pattern": cfgs_model.get("attn", None),  # Pass attn subconfig
            "use_proprio": use_proprio,  # Computed derived value
            "use_action": use_action,  # Computed derived value
        }
    )
    predictor, encoder, action_encoder, proprio_encoder = init_video_model(**model_kwargs)
    encoder_manifest = {
        "encoder_type": cfgs_model["visual_encoder"].get("enc_type"),
        "encoder_version": cfgs_model["visual_encoder"].get("enc_version"),
        "feature_key": getattr(encoder, "feature_key", None),
        "patch_size": int(encoder.patch_size),
        "frozen": bool(freeze_encoder),
        "repo_revision": getattr(encoder, "repo_revision", None),
        "weights_sha256": getattr(encoder, "weights_sha256", None),
        "weights_filename": (
            os.path.basename(encoder.weights_path) if getattr(encoder, "weights_path", None) else None
        ),
    }
    if checkpointing_enabled and cfgs_model["visual_encoder"].get("enc_version", "").startswith("dinov3"):
        for required_provenance in ("repo_revision", "weights_sha256"):
            if not encoder_manifest[required_provenance]:
                raise RuntimeError(f"DINOv3 {required_provenance} is required for research checkpoints")

    heads = {}
    if train_heads or pretrain_dec_path is not None:
        if "image_head" in heads_architectures:
            image_head_type = heads_architectures["image_head"]["kind"]
            if image_head_type is not None and image_head_type.lower() != "none":
                if image_head_type == "vit":
                    decoder = WorldModelViTImageHead(
                        head_config=dict(heads_architectures["image_head"]["config"]),
                        inverse_transform=inverse_transform,
                        device=device,
                    )
                heads["image_head"] = decoder
        if "state_head" in heads_architectures:
            state_decoder = WorldModelPoseReadoutHead(
                head_config=dict(heads_architectures["state_head"]["config"]), device=device
            )
            heads["state_head"] = state_decoder
        if "reward_head" in heads_architectures:
            reward_decoder = WorldModelRewardReadoutHead(
                head_config=dict(heads_architectures["reward_head"]["config"]), device=device
            )
            heads["reward_head"] = reward_decoder

    # -- init optimizer and scheduler
    if train_predictor and predictor is not None:
        optimizer, scaler, scheduler, wd_scheduler = init_opt(
            predictor=predictor,
            action_encoder=action_encoder,
            proprio_encoder=proprio_encoder,
            encoder=encoder,
            freeze_encoder=freeze_encoder,
            **cfgs_opt["transition_model"],
        )
        clip_grad = cfgs_opt["transition_model"]["clip_grad"]
        use_radamw = cfgs_opt["transition_model"]["use_radamw"]
        def rollout_sampling_probability(global_update):
            """Serializable rollout-sampling schedule derived only from update."""

            progress = min(1.0, max(0.0, global_update / float(max(1, ipe * num_epochs))))
            if sampling_scheduler_type == "linear":
                return sampling_scheduler_start + progress * (
                    sampling_scheduler_end - sampling_scheduler_start
                )
            if sampling_scheduler_type == "exponential":
                if sampling_scheduler_start == 0.0:
                    return 0.0
                return sampling_scheduler_start * (
                    sampling_scheduler_end / sampling_scheduler_start
                ) ** progress
            if sampling_scheduler_type == "sigmoid":
                return sampling_scheduler_start + (sampling_scheduler_end - sampling_scheduler_start) * (
                    1 / (1 + np.exp(-10 * (progress - 0.5)))
                )
            raise ValueError(f"Unknown rollout sampling scheduler: {sampling_scheduler_type}")
    else:
        optimizer, scaler, scheduler, wd_scheduler, clip_grad, use_radamw = None, None, None, None, None, None
    if train_heads:
        for name, head in heads.items():
            head.init_opt(**dict(cfgs_opt["heads"][name]))

    start_epoch = 0
    pending_resume_rng_bundle = None
    run_lineage = []
    # -- load training checkpoint
    if load_model:
        continuation_resume = resume_path is not None and os.path.realpath(load_path) == os.path.realpath(resume_path)
        raw_checkpoint = fetch_checkpoint(load_path, device="cpu")
        if is_v2_checkpoint(raw_checkpoint):
            validate_checkpoint_v2(raw_checkpoint)
            saved_encoder = raw_checkpoint["manifest"]["encoder"]
            for provenance_key in ("encoder_version", "repo_revision", "weights_sha256", "patch_size"):
                if saved_encoder.get(provenance_key) != encoder_manifest.get(provenance_key):
                    raise ResumeContractError(
                        f"Encoder provenance mismatch for {provenance_key}: "
                        f"{saved_encoder.get(provenance_key)!r} != {encoder_manifest.get(provenance_key)!r}"
                    )

            # A fork applies only to the external parent.  Once this new stage
            # has published its own ``latest`` role, managed recovery is an
            # exact continuation even though the resolved config retains its
            # lineage declaration.
            fork_from_checkpoint = checkpoint_load_mode == "fork" and not continuation_resume
            if fork_from_checkpoint:
                if cfgs_checkpointing.get("fork_optimizer", "reset") != "reset":
                    raise ValueError(
                        "Only checkpointing.fork_optimizer=reset is supported: a changed dataset or schedule "
                        "must start a new optimizer/scheduler trajectory"
                    )
                parent_manifest = raw_checkpoint["manifest"]
                run_lineage = list(parent_manifest.get("lineage", []))
                run_lineage.append(
                    {
                        "relation": "continued_pretraining_fork",
                        "checkpoint_sha256": sha256_file(load_path),
                        "checkpoint_path": os.path.realpath(load_path),
                        "epoch": int(raw_checkpoint["progress"]["epoch"]),
                        "global_update": int(raw_checkpoint["progress"]["global_update"]),
                        "wandb_run_id": parent_manifest["wandb"]["run_id"],
                        "dataset_fingerprint": parent_manifest["dataset"].get("dataset_fingerprint"),
                        "encoder": parent_manifest["encoder"],
                        "source_git": parent_manifest["git"],
                        "optimizer_policy": "reset",
                    }
                )
            else:
                if not continuation_resume:
                    raise RuntimeError(
                        "Schema-v2 exact continuation checkpoints must be loaded through a managed resume role; "
                        "set checkpointing.load_mode=fork only for an explicitly new pretraining stage"
                    )
                if strict_continuation:
                    validate_resume_contract(raw_checkpoint["manifest"]["resume_contract"], args)
                saved_dataset = raw_checkpoint["manifest"]["dataset"]
                if dataset_manifest is not None and saved_dataset.get(
                    "dataset_fingerprint"
                ) != dataset_manifest.get("dataset_fingerprint"):
                    raise ResumeContractError("Dataset fingerprint changed since the checkpoint was written")
                saved_wandb_id = raw_checkpoint["manifest"]["wandb"]["run_id"]
                if rank == 0 and trainer.use_wandb and trainer.wandb_run_id != saved_wandb_id:
                    raise ResumeContractError(
                        f"W&B run mismatch: checkpoint={saved_wandb_id}, active={trainer.wandb_run_id}"
                    )
                run_lineage = list(raw_checkpoint["manifest"].get("lineage", []))

            state = training_state_from_checkpoint(raw_checkpoint)
            predictor.load_state_dict(clean_state_dict(state["predictor"]), strict=True)
            if action_encoder is not None:
                action_encoder.load_state_dict(clean_state_dict(state["action_encoder"]), strict=True)
            if proprio_encoder is not None:
                proprio_encoder.load_state_dict(clean_state_dict(state["proprio_encoder"]), strict=True)
            if not freeze_encoder:
                encoder.load_state_dict(clean_state_dict(state["encoder"]), strict=True)
            for name, head in heads.items():
                head.model.load_state_dict(clean_state_dict(state["heads"][name]["model"]), strict=True)

            if load_opt_scale_epoch and not fork_from_checkpoint:
                if optimizer is None or state.get("optimizer") is None:
                    raise ResumeContractError("Continuation checkpoint has no transition optimizer state")
                optimizer.load_state_dict(state["optimizer"])
                if scaler is not None:
                    scaler.load_state_dict(state["scaler"])
                scheduler.load_state_dict(state["schedulers"]["lr"])
                wd_scheduler.load_state_dict(state["schedulers"]["weight_decay"])
                for name, head in heads.items():
                    head_state = state["heads"][name]
                    if head.optimizer is not None:
                        head.optimizer.load_state_dict(head_state["optimizer"])
                    if head.scaler is not None:
                        head.scaler.load_state_dict(head_state["scaler"])
                    if head.scheduler is not None:
                        head.scheduler.load_state_dict(head_state["scheduler"])
                    if head.wd_scheduler is not None:
                        head.wd_scheduler.load_state_dict(head_state["weight_decay_scheduler"])

            saved_sampler = raw_checkpoint["sampler"]
            if strict_continuation and not fork_from_checkpoint:
                sampler_contract = saved_sampler["contract"]
                current_sampler_contract = {
                    "world_size": world_size,
                    "sampler_class": type(unsupervised_sampler).__name__,
                    "dataset_length": len(dataset),
                    "batches_per_epoch": len(unsupervised_loader),
                    "iterations_per_epoch": ipe,
                    "batch_size_per_rank": cfgs_loader.get("batch_size"),
                }
                if sampler_contract != current_sampler_contract:
                    raise ResumeContractError(
                        f"Distributed sampler contract changed: {sampler_contract} != {current_sampler_contract}"
                    )
            if fork_from_checkpoint:
                start_epoch = 0
                logger.info(
                    "Forked a schema-v2 checkpoint into a new continued-pretraining stage; "
                    "optimizer, schedulers, progress, sampler, RNG and W&B identity were reset"
                )
            else:
                start_epoch = int(raw_checkpoint["progress"]["epoch"])
                saved_global_update = int(raw_checkpoint["progress"]["global_update"])
                if saved_global_update != start_epoch * ipe:
                    raise ResumeContractError(
                        f"Checkpoint is not an epoch-boundary recovery point: "
                        f"global_update={saved_global_update}, expected={start_epoch * ipe}"
                    )
                # Restore only after DDP, VideoWM, LPIPS and loader iterator
                # construction.  Those constructors may consume CPU/CUDA RNG.
                pending_resume_rng_bundle = raw_checkpoint["rng"]
            del raw_checkpoint
            if not fork_from_checkpoint:
                logger.info(f"Strictly resumed schema-v2 checkpoint at epoch {start_epoch}")
        else:
            del raw_checkpoint
            if continuation_resume and strict_continuation and not cfgs_checkpointing.get(
                "allow_legacy_continuation", False
            ):
                raise ResumeContractError(
                    "Refusing a permissive legacy checkpoint as an exact continuation; "
                    "use it as pretrained weights or explicitly allow_legacy_continuation"
                )
            # Legacy/released checkpoints are intentionally a weights-transfer path.
            load_heads = heads and not pretrain_dec_path and resume
            logger.info(f"Load legacy heads: {load_heads}")
            (
                predictor,
                action_encoder,
                proprio_encoder,
                heads,
                optimizer,
                scaler,
                start_epoch,
            ) = load_checkpoint(
                r_path=load_path,
                predictor=predictor,
                action_encoder=action_encoder,
                proprio_encoder=proprio_encoder,
                heads=heads,
                opt=optimizer,
                scaler=scaler,
                load_opt_scale_epoch=load_opt_scale_epoch,
                load_heads=load_heads,
                train_heads=train_heads,
                train_predictor=train_predictor,
                strict_weights=strict_checkpoint_weights,
            )
            if load_opt_scale_epoch and scheduler is not None and wd_scheduler is not None:
                for _ in range(start_epoch * ipe):
                    scheduler.step()
                    wd_scheduler.step()
        if rollout_only_eval_mode:
            # A released legacy checkpoint may carry an arbitrary training
            # epoch.  Qualification is a single independent evaluation pass.
            start_epoch = 0
        elif light_eval_only_mode:
            start_epoch -= 1

    # Load pretrained heads from pretrain_dec_path
    if pretrain_dec_path is not None:
        for name, head in heads.items():
            if new_path_heads.get(name, True):
                head_path = pretrain_dec_path[name].removesuffix(".pth.tar") + "_" + name + ".pth.tar"
                head.load_checkpoint(head_path)
                logger.info(f"loaded pretrained head named {name}")
            else:
                checkpoint = torch.load(pretrain_dec_path[name], map_location=torch.device("cpu"))
                epoch = checkpoint["epoch"]
                pretrained_dict = clean_state_dict(checkpoint[name])
                msg = head.model.load_state_dict(pretrained_dict, strict=False)
                logger.info(f"loaded pretrained head named {name} from epoch {epoch} with msg: {msg}")
                del checkpoint

    # DDP wrapping after loading state_dicts
    if not freeze_encoder:
        encoder = DDP(encoder, static_graph=False, find_unused_parameters=False)
    if train_predictor:
        if action_encoder is not None:
            action_encoder = DDP(action_encoder, static_graph=False, find_unused_parameters=False)
        if proprio_encoder is not None:
            proprio_encoder = DDP(proprio_encoder, static_graph=False, find_unused_parameters=False)
        predictor = DDP(predictor, static_graph=False, find_unused_parameters=False)
    for name in heads.keys():
        heads[name].model = DDP(heads[name].model, static_graph=False, find_unused_parameters=False)

    # Prepare VideoWM kwargs from config
    wm_kwargs = {
        "device": device,
        # Model components
        "encoder": encoder,
        "predictor": predictor,
        "action_encoder": action_encoder,
        "proprio_encoder": proprio_encoder,
        # Computed dimensions
        "action_dim": model_action_dim,
        "proprio_dim": model_proprio_dim,
        "use_proprio": use_proprio,
        "use_action": use_action,
        # From cfgs_model (pass directly from config)
        "action_tokens": action_tokens,
        "proprio_tokens": proprio_tokens,
        "grid_size": cfgs_model.get("grid_size", 14),
        "tubelet_size_enc": cfgs_model.get("tubelet_size_enc", 2),
        "action_conditioning": cfgs_model.get("action_conditioning", "token"),
        "proprio_encoding": cfgs_model.get("proprio_encoding", "feature"),
        "enc_type": cfgs_model["visual_encoder"].get("enc_type", "vjepa"),
        "pred_type": cfgs_model["predictor"].get("pred_type", "dino_wm"),
        "action_encoder_inpred": cfgs_model["action_encoder"].get("action_encoder_inpred", False),
        "proprio_encoder_inpred": cfgs_model["proprio_encoder"].get("proprio_encoder_inpred", False),
        **cfgs_wm_encoding,
        # From cfgs_data
        "action_skip": cfgs_data.get("action_skip", 1),
        "frameskip": cfgs_data.get("frameskip", 1),
        "img_size": cfgs_data.get("img_size", 256),
        # Heads
        "heads": heads,
        # Optimization
        "scaler": scaler,
        "optimizer": optimizer,
        "clip_grad": clip_grad,
        "mixed_precision": mixed_precision,
        "use_radamw": use_radamw,
        # Loss config (pass subconfig directly)
        "cfgs_loss": cfgs_loss,
    }
    world_model = VideoWM(**wm_kwargs)

    # -- Initialize LPIPS once for evaluation
    lpips = lpips_lib.LPIPS(net="vgg").eval().to(device)

    def _module_state_dict(module):
        if module is None:
            return None
        return (module.module if isinstance(module, DDP) else module).state_dict()

    def _gather_sampler_state(completed_epoch):
        local_state = {
            "rank": rank,
            "sampler_epoch": completed_epoch,
            "sampler_seed": getattr(unsupervised_sampler, "seed", seed),
            "loader_generator": (
                unsupervised_loader.generator.get_state()
                if getattr(unsupervised_loader, "generator", None) is not None
                else None
            ),
            "dataset_rng": dataset.rng.get_state() if hasattr(dataset, "rng") else None,
        }
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            states = [None] * world_size
            torch.distributed.all_gather_object(states, local_state)
        else:
            states = [local_state]
        return {
            "contract": {
                "world_size": world_size,
                "sampler_class": type(unsupervised_sampler).__name__,
                "dataset_length": len(dataset),
                "batches_per_epoch": len(unsupervised_loader),
                "iterations_per_epoch": ipe,
                "batch_size_per_rank": cfgs_loader.get("batch_size"),
            },
            "rank_states": {str(item["rank"]): item for item in states},
            "iteration_in_epoch": 0,
        }

    def save_checkpoint(epoch, path=None, promotion_metrics=None, pending_planning=False):
        """Save an epoch-boundary continuation state; all ranks must call."""

        global_update = epoch * ipe
        if checkpoint_manager is None:
            if rank != 0:
                return None
            save_dict = {
                "predictor": _module_state_dict(world_model.predictor),
                "opt": optimizer.state_dict() if optimizer is not None else None,
                "scaler": None if scaler is None else scaler.state_dict(),
                "epoch": epoch,
                "global_update": global_update,
            }
            if world_model.action_encoder is not None:
                save_dict["action_encoder"] = _module_state_dict(world_model.action_encoder)
            if world_model.proprio_encoder is not None:
                save_dict["proprio_encoder"] = _module_state_dict(world_model.proprio_encoder)
            atomic_torch_save(save_dict, path)
            return None

        rng_bundle = gather_rng_states()
        sampler_bundle = _gather_sampler_state(epoch)
        reference_dict = None
        if rank == 0:
            head_states = {}
            for name, head in world_model.heads.items():
                head_states[name] = {
                    "model": _module_state_dict(head.model),
                    "optimizer": head.optimizer.state_dict() if head.optimizer is not None else None,
                    "scaler": head.scaler.state_dict() if head.scaler is not None else None,
                    "scheduler": head.scheduler.state_dict() if head.scheduler is not None else None,
                    "weight_decay_scheduler": (
                        head.wd_scheduler.state_dict() if head.wd_scheduler is not None else None
                    ),
                }
            training_state = {
                "predictor": _module_state_dict(world_model.predictor),
                "action_encoder": _module_state_dict(world_model.action_encoder),
                "proprio_encoder": _module_state_dict(world_model.proprio_encoder),
                "encoder": _module_state_dict(world_model.encoder) if not freeze_encoder else None,
                "heads": head_states,
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "schedulers": {
                    "lr": scheduler.state_dict() if scheduler is not None else None,
                    "weight_decay": wd_scheduler.state_dict() if wd_scheduler is not None else None,
                    "rollout_sampling": {
                        "type": sampling_scheduler_type,
                        "start": sampling_scheduler_start,
                        "end": sampling_scheduler_end,
                        "global_update": global_update,
                    },
                },
            }
            manifest = create_checkpoint_manifest(
                resolved_config=args,
                dataset_manifest=dataset_manifest,
                encoder_manifest=encoder_manifest,
                wandb_run_id=trainer.wandb_run_id,
                promotion_metrics=promotion_metrics or {},
                lineage=run_lineage,
                require_git_manifest=cfgs_checkpointing.get("require_git_manifest", False),
            )
            checkpoint = build_checkpoint_v2(
                training_state=training_state,
                epoch=epoch,
                global_update=global_update,
                rng_state=rng_bundle,
                sampler_state=sampler_bundle,
                manifest=manifest,
                legacy_fields={
                    "predictor": training_state["predictor"],
                    "action_encoder": training_state["action_encoder"],
                    "proprio_encoder": training_state["proprio_encoder"],
                    "opt": training_state["optimizer"],
                    "scaler": training_state["scaler"],
                    "epoch": epoch,
                    "global_update": global_update,
                },
            )
            reference = checkpoint_manager.save(
                checkpoint,
                global_update=global_update,
                epoch=epoch,
                metadata={
                    "dataset_fingerprint": dataset_manifest.get("dataset_fingerprint"),
                    "encoder_weights_sha256": encoder_manifest.get("weights_sha256"),
                    "wandb_run_id": trainer.wandb_run_id,
                    "promotion_metrics": promotion_metrics or {},
                },
                pending=pending_planning,
            )
            checkpoint_manager.promote("latest", reference)
            reference_dict = reference.to_dict()
            if promotion_metrics and "best_rollout" in promotion_metrics:
                rollout_metric = promotion_metrics["best_rollout"]
                promoted = checkpoint_manager.promote(
                    "best_rollout",
                    reference,
                    metric_name=rollout_metric["metric_name"],
                    metric_value=rollout_metric["metric_value"],
                    mode=rollout_metric.get("mode", "min"),
                    metrics=rollout_metric,
                    tie_break="older",
                )
                if promoted and trainer.use_wandb:
                    wandb.run.summary["best_rollout/checkpoint_id"] = reference.object_id
                    wandb.run.summary["best_rollout/value"] = rollout_metric["metric_value"]
            checkpoint_manager.garbage_collect()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            shared_reference = [reference_dict]
            torch.distributed.broadcast_object_list(shared_reference, src=0)
            reference_dict = shared_reference[0]
        return reference_dict

    logger.info("Initializing loader...")
    train_loader = iter(unsupervised_loader)
    if val_data_iters != [(None, None, None)]:
        val_loader_iters = []
        for vl_dset, vl_traj_dset, vl_loader in val_data_iters:
            val_loader_iters.append(iter(vl_loader))

    if viz_val_data_loader is not None:
        viz_val_loader_iter = iter(viz_val_data_loader)

    if pending_resume_rng_bundle is not None:
        restore_rank_rng_state(
            pending_resume_rng_bundle,
            rank=rank,
            strict_world_size=strict_continuation,
        )
        pending_resume_rng_bundle = None

    def load_clips(sample):
        return sample[0][0].to(device, non_blocking=True), None, None

    def get_batch(train=True, idx=0):
        nonlocal train_loader, val_loader_iters
        if dataset_type == "custom":
            try:
                if train:
                    obs, action, state, reward = next(train_loader)
                else:
                    obs, action, state, reward = next(val_loader_iters[idx])
            except StopIteration as e:
                logger.info(f"Exception {e=}")
                logger.info(f"Exhausted data loaders idx {idx} with {train=}. Refreshing...")
                if train:
                    train_loader = iter(unsupervised_loader)
                    obs, action, state, reward = next(train_loader)
                else:
                    val_loader_iters[idx] = iter(val_data_iters[idx][2])
                    obs, action, state, reward = next(val_loader_iters[idx])
            for k in obs.keys():
                obs[k] = obs[k].to(device, dtype=dtype, non_blocking=True)
            action = action.to(device, dtype=dtype, non_blocking=True)
            state = state.to(device, dtype=dtype, non_blocking=True)
            reward = reward.to(device, dtype=dtype, non_blocking=True)
            return obs, action, state, reward, None, None
        else:
            try:
                if train:
                    sample = next(train_loader)
                else:
                    sample = next(val_loader_iters[idx])
            except StopIteration as e:
                logger.info(f"Exception {e=}")
                logger.info("Exhausted data loaders. Refreshing...")
                if train:
                    train_loader = iter(unsupervised_loader)
                    sample = next(train_loader)
                else:
                    val_loader_iters[idx] = iter(val_data_iters[idx][2])
                    sample = next(val_loader_iters[idx])
            all_clips, all_masks_enc, all_masks_pred = load_clips(sample)
            all_clips = {"visual": all_clips}
            return all_clips, None, None, None, all_masks_enc, all_masks_pred

    def get_viz_batch():
        """Get a visualization batch from the non-distributed validation loader (rank 0 only)"""
        nonlocal viz_val_loader_iter
        if rank != 0 or viz_val_loader_iter is None:
            return None, None, None, None, None, None
        if dataset_type == "custom":
            try:
                obs, action, state, reward = next(viz_val_loader_iter)
            except StopIteration as e:
                logger.info(f"Exception {e=}")
                logger.info("Exhausted viz data loader. Refreshing...")
                viz_val_loader_iter = iter(viz_val_data_loader)
                obs, action, state, reward = next(viz_val_loader_iter)
            for k in obs.keys():
                obs[k] = obs[k].to(device, dtype=torch.float32, non_blocking=True)
            action = action.to(device, dtype=torch.float32, non_blocking=True)
            state = state.to(device, dtype=torch.float32, non_blocking=True)
            reward = reward.to(device, dtype=torch.float32, non_blocking=True)
            return obs, action, state, reward, None, None
        else:
            try:
                sample = next(viz_val_loader_iter)
            except StopIteration as e:
                logger.info(f"Exception {e=}")
                logger.info("Exhausted viz data loader. Refreshing...")
                viz_val_loader_iter = iter(viz_val_data_loader)
                sample = next(viz_val_loader_iter)
            all_clips, all_masks_enc, all_masks_pred = load_clips(sample)
            all_clips = {"visual": all_clips}
            return all_clips, None, None, None, all_masks_enc, all_masks_pred

    # Repair/promote results that may have arrived while a managed training job
    # was preempted.  A recovered synchronous final evaluation must be rerun if
    # it did not publish all results; asynchronous jobs can be drained in place.
    if checkpoint_manager is not None and cfgs_plan_evals and cfgs_plan_evals.get("eval_cfg_paths"):
        planning_registry_path = os.path.join(checkpoint_folder, "planning_results", "registry.json")
        final_drain_timeout = float(cfgs_plan_evals.get("final_drain_timeout_seconds", 4 * 60 * 60))
        final_drain_interval = float(cfgs_plan_evals.get("final_drain_poll_interval_seconds", 10.0))
        final_drain_required = bool(cfgs_plan_evals.get("final_drain_required", True))
        recovery_report = _run_rank0_planning_controller(
            rank,
            lambda: poll_planning_evaluations(planning_registry_path, manager=checkpoint_manager),
            "planning-evaluation recovery poll",
        )
        recovery_plan = [None]

        def build_recovery_plan():
            registry = load_planning_evaluation_registry(planning_registry_path)
            pending_candidates = checkpoint_manager._read_pending()["candidates"]
            registered_pending = set(recovery_report["pending_checkpoint_ids"] if recovery_report else [])
            orphaned = set(pending_candidates) - set(registry["checkpoints"])
            if cfgs_plan_evals.get("separate", True):
                never_launched = {
                    checkpoint_id
                    for checkpoint_id in registered_pending
                    if registry["checkpoints"][checkpoint_id]["launch_state"] == "registered"
                }
                relaunch_ids = orphaned | never_launched
            else:
                # Synchronous workers die with the training job, so every
                # incomplete registered checkpoint must be rerun after recovery.
                relaunch_ids = orphaned | registered_pending
            missing_references = sorted(relaunch_ids - set(pending_candidates))
            if missing_references:
                raise RuntimeError(
                    f"planning registry references checkpoints that are not pinned: {missing_references}"
                )
            references = [pending_candidates[item]["checkpoint"] for item in relaunch_ids]
            references.sort(key=lambda item: int(item["global_update"]))
            recovery_plan[0] = {
                "references": references,
                "remaining_registered_pending": sorted(registered_pending - relaunch_ids),
            }

        _run_rank0_planning_controller(
            rank,
            build_recovery_plan,
            "planning-evaluation recovery-plan construction",
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(recovery_plan, src=0)

        for recovered_reference in recovery_plan[0]["references"]:
            recovered_epoch = int(recovered_reference.get("epoch") or start_epoch)
            logger.info("Relaunching incomplete planning evaluation: %s", recovered_reference["object_id"])
            launch_planning_evals(
                rank,
                recovered_epoch,
                folder,
                recovered_reference["relative_path"],
                cfgs_plan_evals,
                cfgs_model,
                cfgs_data,
                cfgs_data_aug,
                "",
                world_model=world_model,
                dset=val_traj_dataset,
                preprocessor=preprocessor,
                checkpoint_folder=checkpoint_folder,
                checkpoint_reference=recovered_reference,
                checkpoint_manager=checkpoint_manager,
                wandb_trainer=trainer,
                final_epoch=recovered_epoch >= num_epochs,
            )
            world_model.train()

        if (
            start_epoch >= num_epochs
            and recovery_plan[0]["remaining_registered_pending"]
            and cfgs_plan_evals.get("separate", True)
            and (final_drain_timeout > 0 or final_drain_required)
        ):
            _run_rank0_planning_controller(
                rank,
                lambda: drain_planning_evaluations(
                    planning_registry_path,
                    manager=checkpoint_manager,
                    timeout_seconds=final_drain_timeout,
                    poll_interval_seconds=final_drain_interval,
                    raise_on_timeout=final_drain_required,
                ),
                "final planning-evaluation recovery drain",
            )

    # -- TRAINING LOOP
    if not (plan_only_eval_mode or unroll_decode_eval_only_mode):
        measured_training_flops_per_sample = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        training_wall_started = time.monotonic()
        mfu_accumulator = None
        pantheon_heartbeat = None
        latest_mfu_snapshot = None
        telemetry_provenance_record = None
        validation_has_started = False
        if mfu_enabled and not light_eval_only_mode:
            mfu_accumulator = EffectiveMFUAccumulator(
                total_steps=ipe * num_epochs,
                gpu_count=world_size,
                peak_flops_per_gpu=mfu_peak_dense_tflops * 1e12,
                started_at=training_wall_started,
            )
            if rank == 0 and trainer.use_wandb:
                git_commit = os.environ.get("GIT_COMMIT_HASH", "")
                if not re.fullmatch(r"[0-9a-f]{40}", git_commit):
                    raise RuntimeError("training-v1 telemetry requires immutable GIT_COMMIT_HASH")
                provenance = telemetry_provenance(
                    git_commit=git_commit,
                    world_size=world_size,
                    peak_flops_per_gpu=mfu_peak_dense_tflops * 1e12,
                    timed_wall_coverage=1.0,
                )
                telemetry_provenance_record = dict(provenance)
                wandb.config.update(
                    {
                        "pantheon_telemetry": {
                            **provenance,
                            "formula_id": MFU_FORMULA_ID,
                            "formula_version": MFU_FORMULA_VERSION,
                            "global_batch": int(cfgs_loader.get("batch_size")) * world_size,
                            "gradient_accumulation": 1,
                            "experiment_tag": os.environ.get("EXPERIMENT_TAG"),
                            "wandb_run_id": trainer.wandb_run_id,
                        }
                    },
                    allow_val_change=False,
                )
                pantheon_heartbeat = PantheonWandbHeartbeat(
                    wandb.run,
                    total_steps=ipe * num_epochs,
                    interval_seconds=mfu_heartbeat_interval_seconds,
                )
                pantheon_heartbeat.start()
        for epoch in range(start_epoch, num_epochs):
            epoch_wall_started = time.monotonic()
            logger.info("\n" + "─" * 50)
            logger.info(f"📈 Epoch {epoch + 1}/{num_epochs}")
            logger.info("─" * 50)

            # -- update distributed-data-loader epoch
            unsupervised_sampler.set_epoch(epoch)
            if strict_continuation:
                # Recreate workers/iterator from an epoch-derived seed.  This
                # makes the next completed epoch reproducible after recovery.
                if getattr(unsupervised_loader, "generator", None) is not None:
                    unsupervised_loader.generator.manual_seed(seed + rank + epoch * world_size)
                train_loader = iter(unsupervised_loader)

            loss_meter = AverageMeter()
            gpu_time_meter = AverageMeter()
            wall_time_meter = AverageMeter()

            for itr in range(ipe):
                itr_start_time = time.monotonic()
                flop_measurement = {"counter": None, "before_transition_optimizer": None}
                if quick_debug or light_eval_only_mode:
                    if itr > 5:
                        break

                def step_model(obs, action, state, reward, train=True):
                    rates = defaultdict(float)
                    if train:
                        if train_predictor:
                            rates["info/transition_model/lr"] = scheduler.step()
                            rates["info/transition_model/wd"] = wd_scheduler.step()
                        if train_heads:
                            for name, head in world_model.heads.items():
                                rates[f"info/{name}/lr"] = head.scheduler.step()
                                rates[f"info/{name}/wd"] = head.wd_scheduler.step()
                    else:
                        rates["info/transition_model/lr"] = 0.0
                        rates["info/transition_model/wd"] = 0.0
                        for name, head in world_model.heads.items():
                            rates[f"info/{name}/lr"] = 0.0
                            rates[f"info/{name}/wd"] = 0.0
                    # --

                    # Step 1. Forward
                    total_stats = {}
                    total_transition_loss = 0.0
                    total_head_loss = 0.0
                    if action is not None:
                        total_stats.update(
                            {
                                "act_mean": action.mean(),
                                "act_std": action.std(),
                                "act_min": action.min(),
                                "act_max": action.max(),
                            }
                        )
                    # 1. TRAIN PREDICTOR TO PREDICT ONE STEP IN THE FUTURE USING TEACHER FORCING
                    train_rollout_result = {}
                    parallel_rollout_result = {}
                    with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                        video_features, proprio_features, action_features = world_model.encode(obs, action)
                        if train_predictor and train and predictor is not None:
                            pred_video_features, pred_action_features, pred_proprio_features = (
                                world_model.forward_pred(
                                    video_features,
                                    action_features,
                                    proprio_features,
                                )
                            )
                            predictor_losses = world_model.compute_loss(
                                pred_video_features,
                                pred_proprio_features,
                                video_features,
                                proprio_features,
                                shift=1,
                            )
                        else:
                            pred_video_features, pred_proprio_features = None, None
                            predictor_losses = {}
                    predictor_loss = predictor_losses.get("loss", 0.0) / (rollout_steps + 1)
                    if train and train_predictor and predictor is not None:
                        total_transition_loss += predictor_loss
                    stats = defaultdict(list)
                    for k in predictor_losses:
                        val = predictor_losses[k]
                        if isinstance(val, torch.Tensor):
                            val = val.detach().clone()
                        else:
                            val = torch.tensor(val)
                        stats[k].append(val.unsqueeze(0))
                    # mean over 1 element in list
                    stats = {k: torch.stack(stats[k]).mean(0) for k in stats}
                    for k in stats:
                        for j in range(len(stats[k])):
                            train_rollout_result[f"train_rollout/{k}/{j+1}"] = stats[k][j].item()
                    # 2. TRAIN HEADS ON FEATURES FROM ENCODER
                    if "image_head" in world_model.heads and train_heads:
                        # Encoder produces features of videos normalized by mean and std
                        # So let's keep them for target of decoder loss
                        target_rgb_video = world_model.heads["image_head"].preprocess_rgb(
                            obs["visual"][:, ::tubelet_size_enc]
                        )
                        with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                            encoder_image_losses = world_model.heads["image_head"].compute_loss(
                                video_features,
                                target_rgb_video,
                                global_step=epoch * ipe + itr,
                            )
                        rates["info/image_head/examples_seen"] += video_features.shape[0] * video_features.shape[1]
                        encoder_image_losses = {k: v.mean() for k, v in encoder_image_losses.items()}
                        # the weight is like that because we train the head on encoder features and potentially on predictor too
                        loss = encoder_image_losses["loss"] / (train_heads_on_predictor * rollout_steps + 1)
                        if train:
                            total_head_loss += loss
                            world_model.heads["image_head"].backward(loss)
                        encoder_image_losses = {
                            "encoder_image_" + k: v.item() for k, v in encoder_image_losses.items()
                        }
                        total_stats.update(encoder_image_losses)
                    if "state_head" in world_model.heads and train_heads:
                        with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                            encoder_state_losses = world_model.heads["state_head"].compute_loss(
                                video_features, None, state[:, ::tubelet_size_enc]
                            )
                        rates["info/state_head/examples_seen"] += video_features.shape[0] * video_features.shape[1]
                        encoder_state_losses = {k: v.mean() for k, v in encoder_state_losses.items()}
                        loss = encoder_state_losses["loss"] / (train_heads_on_predictor * rollout_steps + 1)
                        if train:
                            total_head_loss += loss
                            world_model.heads["state_head"].backward(loss)
                        encoder_state_losses = {
                            "encoder_state_" + k: v.item() for k, v in encoder_state_losses.items()
                        }
                        total_stats.update(encoder_state_losses)
                    if "reward_head" in world_model.heads and train_heads:
                        with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                            encoder_reward_losses = world_model.heads["reward_head"].compute_loss(
                                video_features, reward[:, ::tubelet_size_enc]
                            )
                        rates["info/reward_head/examples_seen"] += video_features.shape[0] * video_features.shape[1]
                        encoder_reward_losses = {k: v.mean() for k, v in encoder_reward_losses.items()}
                        loss = encoder_reward_losses["loss"] / (train_heads_on_predictor * rollout_steps + 1)
                        if train:
                            total_head_loss += loss
                            world_model.heads["reward_head"].backward(loss)
                        encoder_reward_losses = {
                            "encoder_reward_" + k: v.item() for k, v in encoder_reward_losses.items()
                        }
                        total_stats.update(encoder_reward_losses)
                    # 3. TRAIN HEADS ON FEATURES FROM PREDICTOR
                    if train_heads_on_predictor and train_heads:
                        if "image_head" in world_model.heads:
                            target_rgb_video = world_model.heads["image_head"].preprocess_rgb(
                                obs["visual"][:, ::tubelet_size_enc]
                            )
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                predictor_image_losses = world_model.heads["image_head"].compute_loss(
                                    pred_video_features[:, :-1].detach(),
                                    target_rgb_video[:, 1:],
                                    global_step=epoch * ipe + itr,
                                )
                            rates["info/image_head/examples_seen"] += pred_video_features.shape[0] * (
                                pred_video_features.shape[1] - 1
                            )
                            predictor_image_losses = {k: v.mean() for k, v in predictor_image_losses.items()}
                            if train:
                                world_model.heads["image_head"].backward(predictor_image_losses["loss"])
                            predictor_image_losses = {
                                "predictor_image_" + k: v.item() for k, v in predictor_image_losses.items()
                            }
                            total_stats.update(predictor_image_losses)
                        if "state_head" in world_model.heads:
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                predictor_state_losses = world_model.heads["state_head"].compute_loss(
                                    video_features, None, state[:, ::tubelet_size_enc]
                                )
                            rates["info/state_head/examples_seen"] += video_features.shape[0] * video_features.shape[1]
                            predictor_state_losses = {k: v.mean() for k, v in predictor_state_losses.items()}
                            loss = predictor_state_losses["loss"] / (train_heads_on_predictor * rollout_steps + 1)
                            if train:
                                world_model.heads["state_head"].backward(predictor_state_losses["loss"])
                            predictor_state_losses = {
                                "predictor_state_" + k: v.item() for k, v in predictor_state_losses.items()
                            }
                            total_stats.update(predictor_state_losses)
                    # 4. TRAIN PREDICTOR ON FURTHER AUTOREGRESSIVE ROLLOUT
                    if rollout_steps > 1 and train and train_predictor:
                        if do_sequential_rollout:
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                if train_rollout_prefixes == "random":
                                    prefixes = torch.randint(video_features.shape[1] - rollout_steps, size=(1,))
                                elif train_rollout_prefixes == "first":
                                    prefixes = [0]
                                elif train_rollout_prefixes == "all":
                                    prefixes = list(range(video_features.shape[1] - rollout_steps))
                                stats = defaultdict(list)
                                for t in prefixes:
                                    rollout_losses, total_rollout_loss, _, _ = world_model.rollout(
                                        video_features=video_features,
                                        pred_video_features=pred_video_features,
                                        proprio_features=proprio_features,
                                        pred_proprio_features=pred_proprio_features,
                                        action_features=action_features,
                                        action_noise=0.0,
                                        # we have `len(prefixes)` loss terms so weight them equally
                                        loss_weight=1.0 / len(prefixes),
                                        rollout_steps=rollout_steps - 1,
                                        rollout_stop_gradient=rollout_stop_gradient,
                                        ctxt_window=ctxt_window_train_rollout,
                                        mode="sequential",
                                        t=t,
                                    )
                                    total_transition_loss += total_rollout_loss
                                    for k in rollout_losses:
                                        stats[k].append(rollout_losses[k])
                                stats = {k: torch.stack(stats[k]).mean(0) for k in stats}
                                for k in stats:
                                    for j in range(len(stats[k])):
                                        train_rollout_result[f"train_rollout/{k}/{j+2}"] = stats[k][j].item()
                        if do_parallel_rollout:
                            gt_prob = rollout_sampling_probability(epoch * ipe + itr)
                            rates["info/transition_model/sampling_gt_prob"] = gt_prob
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                stats = defaultdict(list)
                                rollout_parallel_losses, total_rollout_parallel_loss, _, _ = world_model.rollout(
                                    video_features=video_features,
                                    proprio_features=proprio_features,
                                    action_features=action_features,
                                    pred_video_features=pred_video_features,
                                    pred_proprio_features=pred_proprio_features,
                                    action_noise=0.0,
                                    loss_weight=1.0,
                                    rollout_steps=rollout_steps - 1,
                                    rollout_stop_gradient=rollout_stop_gradient,
                                    ctxt_window=ctxt_window_train_rollout,
                                    mode="parallel",
                                    gt_prob=gt_prob,
                                    prepend_gt=prepend_gt_rollout_parallel,
                                )
                                total_transition_loss += total_rollout_parallel_loss
                                for k in rollout_parallel_losses:
                                    stats[k].append(rollout_parallel_losses[k])
                                stats = {k: torch.stack(stats[k]).mean(0) for k in stats}
                                for k in stats:
                                    for j in range(len(stats[k])):
                                        parallel_rollout_result[f"train_rollout_parallel/{k}/{j+2}"] = stats[k][
                                            j
                                        ].item()
                    total_stats.update(train_rollout_result)
                    total_stats.update(parallel_rollout_result)

                    # Construct a clean losses dict that aggregates all relevant training losses
                    losses = {}
                    losses["predictor_loss"] = (
                        total_transition_loss.item()
                        if isinstance(total_transition_loss, torch.Tensor)
                        else total_transition_loss
                    )
                    losses["head_loss"] = (
                        total_head_loss.item() if isinstance(total_head_loss, torch.Tensor) else total_head_loss
                    )
                    losses["loss"] = losses["predictor_loss"] + losses["head_loss"]

                    grad_stats, optim_stats = {}, {}
                    # 5. OPTIMIZATION STEP
                    # so far, we only computed losses and ran .backward() so accumulate gradients on several objectives. Below is the
                    # only place where we perform an actual optimization step
                    if train:
                        if train_heads:
                            for name in world_model.heads.keys():
                                # If not train_heads, world_model.heads[name].model.module.decoder_embed.weight.grad should be None
                                grad_stats[name], optim_stats[name] = world_model.heads[name].optimization_step()
                        if train_predictor:
                            world_model.backward(total_transition_loss)
                            if flop_measurement["counter"] is not None:
                                # The useful-work numerator includes the frozen
                                # encoder forward and trainable predictor
                                # forward/backward, but excludes optimizer math.
                                flop_measurement["before_transition_optimizer"] = float(
                                    flop_measurement["counter"].get_total_flops()
                                )
                            grad_stats["transition_model"], optim_stats["transition_model"] = (
                                world_model.optimization_step()
                            )
                        for key in list(grad_stats.keys()):
                            grad_stats[f"optim/{key}/grad_norm"] = (
                                grad_stats[key].global_norm if grad_stats[key] is not None else 0.0
                            )
                            del grad_stats[key]
                        for key in list(optim_stats.keys()):
                            optim_stats[f"optim/{key}/first_moment"] = (
                                optim_stats[key].get("exp_avg").avg if optim_stats[key] is not None else 0.0
                            )
                            optim_stats[f"optim/{key}/second_moment"] = (
                                optim_stats[key].get("exp_avg_sq").avg if optim_stats[key] is not None else 0.0
                            )
                            del optim_stats[key]
                    total_stats.update(grad_stats)
                    total_stats.update(optim_stats)
                    total_stats.update(dict(rates))
                    # 6. ONCE IN A WHILE DO LONG-ROLLOUT EVALUATION WITH DATASET ACTIONS OR RANDOM ACTIONS
                    eval_rollout_result = {}
                    image_stats = {}
                    if not train:
                        world_model.eval()

                        @torch.no_grad
                        def val_rollout(
                            video_features,
                            action_features,
                            proprio_features,
                            pred_video_features,
                            pred_proprio_features,
                            gt_obs,
                            gt_state,
                            prefix_rollout_result=None,
                            val_rollout_steps=5,
                            ctxt_window=None,
                        ):
                            """
                            gt_obs:
                                visual: (B, T, C, H, W)
                            """
                            rollout_steps = min(val_rollout_steps, video_features.shape[1] - 1)
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                prefixes = range(video_features.shape[1] - rollout_steps)
                                val_rollout_result = {}

                                # Helper function to run rollout with different action noise levels and decode heads
                                def run_rollout_and_decode(action_noise, rollout_prefix):
                                    """Run rollout with given action noise and decode with image/state heads."""
                                    stats = defaultdict(list)
                                    for t in prefixes:
                                        rollout_losses, _, last_vid_feats, last_prop_feats = world_model.rollout(
                                            video_features=video_features,
                                            pred_video_features=pred_video_features,
                                            proprio_features=proprio_features,
                                            pred_proprio_features=pred_proprio_features,
                                            action_features=action_features,
                                            action_noise=action_noise,
                                            rollout_steps=rollout_steps,
                                            rollout_stop_gradient=True,
                                            debug=action_noise == 0.0,
                                            ctxt_window=ctxt_window,
                                            mode="sequential",
                                            t=t,
                                        )
                                        # last_vid_feats: [B T V H W D]
                                        for k in rollout_losses:
                                            stats[k].append(rollout_losses[k])

                                    image_samples = None
                                    if "image_head" in world_model.heads:
                                        image_samples = world_model.heads["image_head"].decode(
                                            last_vid_feats[:, -rollout_steps - 1 :]
                                        )
                                        for h in range(1, image_samples.shape[1]):
                                            key = f"{rollout_prefix}/lpips/{h}"
                                            if prefix_rollout_result is not None:
                                                key = f"{prefix_rollout_result}/{key}"
                                            v = lpips(
                                                image_samples.squeeze(2)[:, h]
                                                .permute(0, 3, 1, 2)
                                                .to(world_model.device, dtype=torch.float32)
                                                / 255.0,
                                                gt_obs["visual"][:, h * tubelet_size_enc].to(
                                                    world_model.device, dtype=torch.float32
                                                ),
                                            ).mean()
                                            val_rollout_result[key] = v.detach().cpu().item()

                                    if "state_head" in world_model.heads:
                                        target_states = gt_state[:, ::tubelet_size_enc][:, -rollout_steps - 1 :]
                                        state_loss = world_model.heads["state_head"].compute_loss(
                                            last_vid_feats[:, -rollout_steps - 1 :],
                                            None,
                                            target_states,
                                            reduce_mean=False,
                                        )
                                        for k in state_loss:
                                            for j in range(state_loss[k].shape[1]):
                                                key = f"{rollout_prefix}/encoder_state_{k}/{j}"
                                                if prefix_rollout_result is not None:
                                                    key = f"{prefix_rollout_result}/{key}"
                                                val_rollout_result[key] = state_loss[k][:, j].mean().item()

                                    # Aggregate rollout losses
                                    stats = {k: torch.stack(stats[k]).mean(0) for k in stats}
                                    for k in stats:
                                        for j in range(len(stats[k])):
                                            key = f"{rollout_prefix}/{k}/{j+1}"
                                            if prefix_rollout_result is not None:
                                                key = f"{prefix_rollout_result}/{key}"
                                            val_rollout_result[key] = stats[k][j].item()

                                    return image_samples

                                eval_image_samples = run_rollout_and_decode(
                                    action_noise=0.0, rollout_prefix="val_rollout"
                                )

                                # Optionally decode position from ground truth visual features
                                if "state_head" in world_model.heads and data_traj_decode_gt:
                                    target_states = gt_state[:, ::tubelet_size_enc][:, -rollout_steps - 1 :]
                                    decode_gt_state_loss = world_model.heads["state_head"].compute_loss(
                                        video_features[:, -rollout_steps - 1 :], None, target_states, reduce_mean=False
                                    )
                                    for k in decode_gt_state_loss:
                                        for j in range(decode_gt_state_loss[k].shape[1]):
                                            key = f"val_rollout/decode_gt_state_{k}/{j}"
                                            if prefix_rollout_result is not None:
                                                key = f"{prefix_rollout_result}/{key}"
                                            val_rollout_result[key] = decode_gt_state_loss[k][:, j].mean().item()

                                noisy_eval_image_samples = run_rollout_and_decode(
                                    action_noise=0.05, rollout_prefix="noisy_val_rollout"
                                )

                                if "image_head" in world_model.heads:
                                    t = eval_image_samples.shape[1]
                                    b = gt_obs["visual"].shape[0]
                                    # b = gt_obs["visual"][:, ::tubelet_size_enc].shape[0]
                                    b = min(4, b)
                                    rgb_v = inverse_transform(gt_obs["visual"][:, ::tubelet_size_enc].cpu())
                                    rgb_v = (255.0 * rgb_v).clip(0.0, 255.0).to(torch.uint8)
                                    rgb_v = rearrange(rgb_v, "b t (v c) h w -> b t v h w c", c=3)
                                    rgb = torch.stack([eval_image_samples, rgb_v[:, -t:]], dim=2)[:b]
                                    rgb = (
                                        rearrange(
                                            rgb,
                                            "b t e v h w c -> (b v e h) (t w) c",
                                        )
                                        .cpu()
                                        .numpy()
                                    )
                                    rgb_noised = torch.stack([noisy_eval_image_samples, rgb_v[:, -t:]], dim=2)[:b]
                                    rgb_noised = (
                                        rearrange(
                                            rgb_noised,
                                            "b t e v h w c -> (b v e h) (t w) c",
                                        )
                                        .cpu()
                                        .numpy()
                                    )
                                    animation = torch.stack(
                                        [rgb_v[:, -t:], eval_image_samples, noisy_eval_image_samples], dim=2
                                    )[:b]
                                    animation = (
                                        rearrange(
                                            animation,
                                            "b t e v h w c -> t c (b v h) (e w)",
                                        )
                                        .cpu()
                                        .numpy()
                                    )
                                else:
                                    rgb, rgb_noised, animation = None, None, None
                            return rgb, rgb_noised, animation, val_rollout_result

                        if do_data_traj_rollout_eval:
                            rgb, rgb_noised, animation, eval_rollout_result = val_rollout(
                                video_features,
                                action_features,
                                proprio_features,
                                pred_video_features,
                                pred_proprio_features,
                                gt_obs=obs,
                                prefix_rollout_result="data_traj",
                                val_rollout_steps=cfgs_data_traj_rollout_eval.get("data_traj_eval_rollout_steps", 3),
                                ctxt_window=cfgs_data_traj_rollout_eval.get("data_traj_eval_ctxt_window", None),
                                gt_state=state,
                            )
                        if do_energy_landscape_eval:
                            gt_visual = inverse_transform(obs["visual"].cpu())
                            gt_visual = (255.0 * gt_visual).clip(0.0, 255.0).to(torch.uint8)
                            gt_proprio = preprocessor.denormalize_proprios(obs["proprio"].cpu())
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                energy_landscape = world_model.compute_energy_landscape(
                                    video_features,
                                    action_features,
                                    proprio_features,
                                    gt_visual,
                                    gt_proprio,
                                    rollout_steps=cfgs_energy_landscape_eval.get("energy_landscape_rollout_steps", 1),
                                    ctxt_window=cfgs_energy_landscape_eval.get("energy_landscape_ctxt_window", None),
                                    proprio_dim=traj_dataset.proprio_dim,
                                    action_dim=traj_dataset.action_dim,
                                    actions_per_vid_feat=actions_per_vid_feat,
                                    dataset_path=dataset_paths[0],
                                    preprocessor=preprocessor,
                                )
                            image_stats.update(
                                {
                                    "data_traj/energy_landscape": energy_landscape,
                                }
                            )
                        if "image_head" in world_model.heads:
                            if do_data_traj_rollout_eval:
                                image_stats.update(
                                    {
                                        "data_traj/image_rollouts": wandb.Image(rgb),
                                        "data_traj/image_rollouts_noisy_actions": wandb.Image(rgb_noised),
                                        "data_traj/image_animated_rollout": wandb.Video(animation, fps=6),
                                    }
                                )
                        world_model.train()
                    total_stats.update(eval_rollout_result)
                    return (
                        float(losses["loss"]),
                        losses,
                        optim_stats,
                        total_stats,
                        image_stats,
                    )

                # In train mode, image_stats is empty
                if not light_eval_only_mode:
                    obs, action, state, reward, masks_enc, masks_pred = get_batch()
                    measure_mfu_step = (
                        mfu_enabled
                        and measured_training_flops_per_sample is None
                        and epoch >= mfu_measure_epoch
                        and itr == mfu_measure_iteration
                    )
                    if measure_mfu_step:
                        from torch.utils.flop_counter import FlopCounterMode

                        with FlopCounterMode(display=False) as flop_counter:
                            flop_measurement["counter"] = flop_counter
                            (loss, losses, optim_stats, total_stats, image_stats), gpu_etime_ms = gpu_timer(
                                lambda: step_model(obs, action, state, reward, train=True)
                            )
                        measured_step_flops = flop_measurement["before_transition_optimizer"]
                        if measured_step_flops is None:
                            measured_step_flops = float(flop_counter.get_total_flops())
                        measured_batch_size = int(obs["visual"].shape[0])
                        if measured_step_flops <= 0 or measured_batch_size <= 0:
                            raise RuntimeError(
                                "MFU profiling recorded no training FLOPs or an empty batch"
                            )
                        measured_training_flops_per_sample = measured_step_flops / measured_batch_size
                        logger.info(
                            "Measured %.3e training FLOPs/sample for MFU telemetry",
                            measured_training_flops_per_sample,
                        )
                    else:
                        (loss, losses, optim_stats, total_stats, image_stats), gpu_etime_ms = gpu_timer(
                            lambda: step_model(obs, action, state, reward, train=True)
                        )
                    iter_elapsed_time_ms = (time.monotonic() - itr_start_time) * 1000.0
                    if measured_training_flops_per_sample is not None:
                        total_stats.update(
                            training_performance_stats(
                                flops_per_sample=measured_training_flops_per_sample,
                                local_batch_size=int(obs["visual"].shape[0]),
                                world_size=world_size,
                                gpu_elapsed_ms=gpu_etime_ms,
                                wall_elapsed_ms=iter_elapsed_time_ms,
                                peak_dense_tflops=mfu_peak_dense_tflops,
                            )
                        )
                    if mfu_accumulator is not None:
                        mfu_accumulator.record_samples(int(obs["visual"].shape[0]) * world_size)
                        if measured_training_flops_per_sample is not None:
                            mfu_accumulator.set_flops_per_sample(measured_training_flops_per_sample)
                    loss_meter.update(loss)
                    gpu_time_meter.update(gpu_etime_ms)
                    wall_time_meter.update(iter_elapsed_time_ms)
                    if train_csv_logger is None:  # Initialize the logger once
                        train_csv_logger = create_csv_logger(losses, total_stats, train=True)
                else:
                    losses = {}
                    total_stats = {}

                if itr % light_eval_freq == light_eval_freq - 1 and val_loader_iters is not None:
                    validation_has_started = True
                    # image_stats overrides the empty image_stats from the train step at same itr
                    image_stats, eval_losses, eval_total_stats = {}, {}, {}
                    # Then, use non-distributed validation data for visualization on rank 0
                    if val_viz_rank0_loader and rank == 0:
                        viz_obs, viz_action, viz_state, viz_reward, _, _ = get_viz_batch()
                        if viz_obs is not None:
                            with torch.no_grad():
                                (_, _, _, _, viz_img_stats), _ = gpu_timer(
                                    lambda: step_model(viz_obs, viz_action, viz_state, viz_reward, train=False)
                                )
                                if light_eval_only_mode:
                                    image_stats.update(
                                        {f"light_eval_only/epoch-{epoch+1}/{k}": v for k, v in viz_img_stats.items()}
                                    )
                                else:
                                    image_stats.update(viz_img_stats)
                    # First, use distributed validation data for metrics
                    for idx in range(len(val_loader_iters)):
                        obs, action, state, reward, masks_enc, masks_pred = get_batch(train=False, idx=idx)
                        with torch.no_grad():
                            (_, val_losses, _, val_total_stats, val_img_stats), gpu_etime_ms = gpu_timer(
                                lambda: step_model(obs, action, state, reward, train=False)
                            )
                            if not light_eval_only_mode:
                                eval_losses.update({f"eval_data/load-{idx}/{k}": v for k, v in val_losses.items()})
                                eval_total_stats.update(
                                    {f"eval_data/load-{idx}/{k}": v for k, v in val_total_stats.items()}
                                )
                            # Only use distributed batch images if we don't have visualization images
                            if not val_viz_rank0_loader:
                                prefixed_stats = {f"load-{idx}/{k}": v for k, v in val_img_stats.items()}
                                if light_eval_only_mode:
                                    image_stats.update(
                                        {f"light_eval_only/epoch-{epoch+1}/{k}": v for k, v in prefixed_stats.items()}
                                    )
                                else:
                                    image_stats.update(prefixed_stats)
                    if eval_csv_logger is None:  # Initialize the logger once
                        eval_csv_logger = create_csv_logger(eval_losses, eval_total_stats, train=False)
                else:
                    eval_losses = {}
                    eval_total_stats = {}

                if mfu_accumulator is not None and (
                    (itr + 1) % mfu_telemetry_interval_steps == 0 or (epoch == num_epochs - 1 and itr == ipe - 1)
                ):
                    local_step_time_s = max(1e-9, time.monotonic() - itr_start_time)
                    local_elapsed_e2e_s = max(1e-9, time.monotonic() - training_wall_started)
                    timing = torch.tensor(
                        [local_step_time_s, local_elapsed_e2e_s],
                        dtype=torch.float64,
                        device=device,
                    )
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        torch.distributed.all_reduce(timing, op=torch.distributed.ReduceOp.MAX)
                    latest_mfu_snapshot = mfu_accumulator.snapshot(
                        active_step=epoch * ipe + itr + 1,
                        step_time_s=float(timing[0].item()),
                        elapsed_e2e_s=float(timing[1].item()),
                        timed_wall_coverage=1.0,
                    )
                    total_stats.update(latest_mfu_snapshot.history())
                    if rank == 0 and pantheon_heartbeat is not None:
                        pantheon_heartbeat.update(latest_mfu_snapshot)

                # -- Logging
                def log_stats():
                    trainer.log(epoch, itr, losses, total_stats, eval_losses, eval_total_stats, image_stats)
                    if not light_eval_only_mode:
                        log_values = [epoch + 1, itr, loss, gpu_etime_ms, iter_elapsed_time_ms]
                        for key in train_csv_logger_columns[5:]:
                            if key in losses:
                                value = losses[key]
                                log_values.append(value.item() if isinstance(value, torch.Tensor) else value)
                            elif key in total_stats:
                                value = total_stats[key]
                                log_values.append(value.item() if isinstance(value, torch.Tensor) else value)
                            else:
                                log_values.append(0.0)
                        train_csv_logger.log(*log_values)
                        if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                            logger.info(
                                "[%d, %5d] "
                                "[mem: %.2e] "
                                "[gpu: %.1f ms]"
                                "[wall: %.1f ms]"
                                % (
                                    epoch + 1,
                                    itr,
                                    torch.cuda.max_memory_allocated() / 1024.0**2,
                                    gpu_time_meter.avg,
                                    wall_time_meter.avg,
                                )
                            )
                    if itr % light_eval_freq == light_eval_freq - 1:
                        log_values = [epoch + 1, itr]
                        for key in eval_csv_logger_columns[2:]:
                            if key in eval_losses:
                                value = eval_losses[key]
                                log_values.append(value.item() if isinstance(value, torch.Tensor) else value)
                            elif key in eval_total_stats:
                                log_values.append(eval_total_stats[key])
                            else:
                                log_values.append(0.0)
                        eval_csv_logger.log(*log_values)

                log_stats()
                if not light_eval_only_mode:
                    assert not np.isnan(loss), "loss is nan"
            logger.info("avg. loss %.3f" % loss_meter.avg)

            # Publish recovery state immediately at the completed epoch boundary.
            # Rollout/planning evaluation happens only after this durable point, so
            # expensive evaluation can never extend the useful-work loss window.
            promotion_metrics = {}
            planning_due = bool(cfgs_plan_evals and cfgs_plan_evals.get("eval_cfg_paths")) and (
                (epoch % eval_freq == 0) or epoch == (num_epochs - 1)
            )
            saved_reference = None
            checkpoint_time_s = 0.0
            epoch_boundary_time_s = 0.0
            if not light_eval_only_mode and (
                (epoch + 1) % checkpoint_save_every == 0 or epoch == (num_epochs - 1)
            ):
                checkpoint_started = time.monotonic()
                saved_reference = save_checkpoint(
                    epoch + 1,
                    latest_path,
                    promotion_metrics=None,
                    pending_planning=planning_due,
                )
                timing = torch.tensor(
                    [
                        max(1e-9, time.monotonic() - checkpoint_started),
                        max(1e-9, time.monotonic() - epoch_wall_started),
                        max(1e-9, time.monotonic() - training_wall_started),
                    ],
                    dtype=torch.float64,
                    device=device,
                )
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(timing, op=torch.distributed.ReduceOp.MAX)
                checkpoint_time_s = float(timing[0].item())
                epoch_boundary_time_s = float(timing[1].item())
                within_loss_budget = epoch_boundary_time_s <= checkpoint_loss_budget_seconds
                if rank == 0:
                    timing_path = os.path.join(
                        checkpoint_folder, "run_metadata", "epoch_boundary_timing.json"
                    )
                    previous_max = 0.0
                    if os.path.exists(timing_path):
                        with open(timing_path, "r", encoding="utf-8") as stream:
                            previous_max = float(json.load(stream).get("max_epoch_boundary_seconds", 0.0))
                    atomic_json_dump(
                        {
                            "schema_version": 1,
                            "kind": "jepa-wm-epoch-boundary-checkpoint-timing",
                            "epoch_boundary_only": True,
                            "epoch": epoch + 1,
                            "global_update": (epoch + 1) * ipe,
                            "checkpoint_seconds": checkpoint_time_s,
                            "epoch_boundary_seconds": epoch_boundary_time_s,
                            "max_epoch_boundary_seconds": max(previous_max, epoch_boundary_time_s),
                            "loss_budget_seconds": checkpoint_loss_budget_seconds,
                            "within_loss_budget": within_loss_budget,
                        },
                        timing_path,
                    )
                    if trainer.use_wandb:
                        wandb.log(
                            {
                                "train/global_step": (epoch + 1) * ipe,
                                "perf/checkpoint_time_s": checkpoint_time_s,
                                "perf/epoch_boundary_interval_s": epoch_boundary_time_s,
                                "perf/checkpoint_loss_budget_s": checkpoint_loss_budget_seconds,
                            },
                            step=(epoch + 1) * ipe,
                        )
                    if (
                        mfu_accumulator is not None
                        and latest_mfu_snapshot is not None
                        and telemetry_provenance_record is not None
                        and trainer.use_wandb
                    ):
                        latest_mfu_snapshot = mfu_accumulator.snapshot(
                            active_step=(epoch + 1) * ipe,
                            step_time_s=latest_mfu_snapshot.step_time_s,
                            elapsed_e2e_s=float(timing[2].item()),
                            timed_wall_coverage=1.0,
                        )
                        if pantheon_heartbeat is not None:
                            pantheon_heartbeat.update(latest_mfu_snapshot)
                        telemetry_document = telemetry_snapshot_document(
                            latest_mfu_snapshot,
                            wandb_url=wandb.run.url,
                            provenance=telemetry_provenance_record,
                            validation_started=validation_has_started,
                        )
                        telemetry_directory = os.path.join(checkpoint_folder, "run_metadata")
                        atomic_json_dump(
                            telemetry_document,
                            os.path.join(
                                telemetry_directory,
                                f"training_v1_step-{(epoch + 1) * ipe:012d}.json",
                            ),
                        )
                        atomic_json_dump(
                            telemetry_document,
                            os.path.join(telemetry_directory, "training_v1_snapshot.json"),
                        )
                if not within_loss_budget:
                    raise RuntimeError(
                        f"Epoch {epoch + 1} plus checkpoint took {epoch_boundary_time_s:.1f}s, "
                        f"exceeding the {checkpoint_loss_budget_seconds:.1f}s Pantheon work-loss budget"
                    )
                if checkpoint_manager is None and rank == 0:
                    if save_every_freq > 0 and epoch % save_every_freq == 0:
                        save_every_file = pref_tag + f"e{epoch}.{latest_format}"
                        save_every_path = os.path.join(checkpoint_folder, save_every_file)
                        save_checkpoint(epoch + 1, save_every_path)

            # -- Deterministic held-out rollout promotion
            rollout_promotion_cfg = cfgs_checkpointing.get("rollout_promotion", {})
            rollout_promotion_due = (
                checkpointing_enabled
                and rollout_promotion_cfg.get("enabled", False)
                and (
                    (epoch + 1) % int(rollout_promotion_cfg.get("every_epochs", 1)) == 0
                    or epoch == (num_epochs - 1)
                )
            )
            if rollout_promotion_due:
                loader_index = int(rollout_promotion_cfg.get("loader_index", 0))
                if not val_loader_iters or loader_index >= len(val_loader_iters):
                    raise RuntimeError(f"No validation loader {loader_index} for best_rollout promotion")
                promotion_loader = val_data_iters[loader_index][2]
                if hasattr(promotion_loader.sampler, "set_epoch"):
                    promotion_loader.sampler.set_epoch(int(rollout_promotion_cfg.get("sampler_epoch", 0)))
                if getattr(promotion_loader, "generator", None) is not None:
                    promotion_loader.generator.manual_seed(
                        int(rollout_promotion_cfg.get("seed", seed + 50_000 + loader_index))
                    )
                val_loader_iters[loader_index] = iter(promotion_loader)
                max_batches = rollout_promotion_cfg.get("max_batches")
                num_batches = len(promotion_loader) if max_batches is None else min(int(max_batches), len(promotion_loader))
                max_horizon = min(
                    int(cfgs_data_traj_rollout_eval.get("data_traj_eval_rollout_steps", 1)),
                    int(cfgs_validation.get("num_frames_val", 2)) - 1,
                )
                horizons = rollout_promotion_cfg.get("horizons", list(range(1, max_horizon + 1)))
                metric_base = rollout_promotion_cfg.get(
                    "metric_base", "data_traj/val_rollout/visual_l2_loss"
                )
                metric_keys = [f"{metric_base}/{int(horizon)}" for horizon in horizons]
                metric_sums = {key: 0.0 for key in metric_keys}
                metric_count = 0
                for _ in range(num_batches):
                    p_obs, p_action, p_state, p_reward, _, _ = get_batch(train=False, idx=loader_index)
                    with torch.no_grad():
                        _, _, _, p_stats, _ = step_model(p_obs, p_action, p_state, p_reward, train=False)
                    missing_metrics = [key for key in metric_keys if key not in p_stats]
                    if missing_metrics:
                        raise RuntimeError(
                            f"Rollout promotion metrics missing: {missing_metrics}; "
                            f"available={sorted(p_stats)}"
                        )
                    batch_count = int(p_obs["visual"].shape[0])
                    for key in metric_keys:
                        metric_sums[key] += float(p_stats[key]) * batch_count
                    metric_count += batch_count
                reduced = torch.tensor(
                    [metric_sums[key] for key in metric_keys] + [metric_count],
                    dtype=torch.float64,
                    device=device,
                )
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
                total_count = int(reduced[-1].item())
                if total_count <= 0:
                    raise RuntimeError("Rollout promotion evaluated zero examples")
                horizon_metrics = {
                    key: reduced[index].item() / total_count for index, key in enumerate(metric_keys)
                }
                aggregation = rollout_promotion_cfg.get("aggregation", "mean")
                if aggregation != "mean":
                    raise ValueError("checkpointing.rollout_promotion.aggregation must be 'mean'")
                primary_key = rollout_promotion_cfg.get(
                    "metric_name",
                    f"{metric_base}/mean_h" + "_h".join(str(int(horizon)) for horizon in horizons),
                )
                primary_value = float(np.mean([horizon_metrics[key] for key in metric_keys]))
                if not np.isfinite(primary_value):
                    raise RuntimeError(f"Non-finite rollout promotion metric: {primary_value}")
                promotion_metrics["best_rollout"] = {
                    "metric_name": primary_key,
                    "metric_value": primary_value,
                    "mode": "min",
                    "count": total_count,
                    "loader_index": loader_index,
                    "horizons": horizon_metrics,
                    "aggregation": aggregation,
                    "sampler_epoch": int(rollout_promotion_cfg.get("sampler_epoch", 0)),
                    "seed": int(rollout_promotion_cfg.get("seed", seed + 50_000 + loader_index)),
                }

                if not rollout_only_eval_mode:
                    if checkpoint_manager is None or saved_reference is None:
                        raise RuntimeError(
                            "Rollout promotion requires the immutable epoch-boundary checkpoint"
                        )

                    def promote_saved_rollout_checkpoint():
                        metric = promotion_metrics["best_rollout"]
                        return checkpoint_manager.promote(
                            "best_rollout",
                            saved_reference,
                            metric_name=metric["metric_name"],
                            metric_value=metric["metric_value"],
                            mode=metric["mode"],
                            metrics=metric,
                            tie_break="older",
                        )

                    rollout_promoted = _run_rank0_planning_controller(
                        rank,
                        promote_saved_rollout_checkpoint,
                        "best-rollout checkpoint promotion",
                    )
                    if rank == 0 and trainer.use_wandb and rollout_promoted:
                        wandb.run.summary["best_rollout/checkpoint_id"] = saved_reference["object_id"]
                        wandb.run.summary["best_rollout/value"] = primary_value

                if rollout_only_eval_mode:
                    if load_path is None:
                        raise RuntimeError("rollout-only qualification requires a loaded checkpoint")

                    def publish_rollout_qualification():
                        existing_alias = checkpoint_manager.read_alias("latest", verify=True)
                        if existing_alias is not None:
                            reference = existing_alias["checkpoint"]
                        else:
                            registered = checkpoint_manager.register_existing(
                                load_path,
                                global_update=0,
                                epoch=None,
                                metadata={
                                    "artifact_kind": "released_qualification_checkpoint",
                                    "source_path": os.path.realpath(load_path),
                                    "wandb_run_id": trainer.wandb_run_id,
                                },
                            )
                            reference = registered.to_dict()
                            checkpoint_manager.promote("latest", reference)

                        metric = promotion_metrics["best_rollout"]
                        promoted = checkpoint_manager.promote(
                            "best_rollout",
                            reference,
                            metric_name=metric["metric_name"],
                            metric_value=metric["metric_value"],
                            mode=metric["mode"],
                            metrics=metric,
                        )
                        promotion_config_sha256 = hashlib.sha256(
                            json.dumps(
                                rollout_promotion_cfg,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        ).hexdigest()
                        result = {
                            "schema_version": 1,
                            "status": "complete",
                            "artifact_kind": "released_rollout_qualification",
                            "checkpoint": reference,
                            "dataset_fingerprint": (dataset_manifest or {}).get("dataset_fingerprint"),
                            "promotion_config_sha256": promotion_config_sha256,
                            "promotion_config": rollout_promotion_cfg,
                            "selection": metric,
                            "promoted": bool(promoted),
                            "wandb_run_id": trainer.wandb_run_id,
                        }
                        result_path = os.path.join(
                            checkpoint_folder,
                            "qualification_results",
                            "released_rollout.json",
                        )
                        atomic_json_dump(result, result_path)
                        return result

                    qualification_result = _run_rank0_planning_controller(
                        rank,
                        publish_rollout_qualification,
                        "released rollout qualification publication",
                    )
                    if rank == 0 and trainer.use_wandb:
                        wandb.run.summary["qualification/released_rollout/status"] = "complete"
                        wandb.run.summary["qualification/released_rollout/checkpoint_sha256"] = (
                            qualification_result["checkpoint"]["sha256"]
                        )
                        wandb.run.summary["qualification/released_rollout/value"] = primary_value
                if rank == 0 and trainer.use_wandb:
                    wandb.log(
                        {
                            "train/global_step": (epoch + 1) * ipe,
                            "promotion/rollout/primary": primary_value,
                            "promotion/rollout/count": total_count,
                            **{f"promotion/rollout/{key}": value for key, value in horizon_metrics.items()},
                        },
                        step=(epoch + 1) * ipe,
                    )

            # -- Launch Planning Eval
            if not light_eval_only_mode:
                if planning_due:
                    if saved_reference is not None:
                        checkpoint = saved_reference["relative_path"]
                    elif save_every_freq > 0:
                        checkpoint = (
                            pref_tag + f"latest.{latest_format}"
                            if epoch == (num_epochs - 1)
                            else pref_tag + f"e{epoch}.{latest_format}"
                        )
                    else:
                        checkpoint = pref_tag + f"latest.{latest_format}"
                    launch_planning_evals(
                        rank,
                        epoch + 1,
                        folder,
                        checkpoint,
                        cfgs_plan_evals,
                        cfgs_model,
                        cfgs_data,
                        cfgs_data_aug,
                        "",
                        world_model=world_model,
                        dset=val_traj_dataset,
                        preprocessor=preprocessor,
                        checkpoint_folder=checkpoint_folder,
                        checkpoint_reference=saved_reference,
                        checkpoint_manager=checkpoint_manager,
                        wandb_trainer=trainer,
                        final_epoch=epoch == (num_epochs - 1),
                    )
                    world_model.train()
        if mfu_accumulator is not None and latest_mfu_snapshot is not None:
            final_timing = torch.tensor(
                [latest_mfu_snapshot.step_time_s, max(1e-9, time.monotonic() - training_wall_started)],
                dtype=torch.float64,
                device=device,
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(final_timing, op=torch.distributed.ReduceOp.MAX)
            latest_mfu_snapshot = mfu_accumulator.snapshot(
                active_step=min(ipe * num_epochs, max(start_epoch * ipe, latest_mfu_snapshot.active_step)),
                step_time_s=float(final_timing[0].item()),
                elapsed_e2e_s=float(final_timing[1].item()),
                timed_wall_coverage=1.0,
            )
            if rank == 0 and trainer.use_wandb:
                if pantheon_heartbeat is not None:
                    pantheon_heartbeat.update(latest_mfu_snapshot)
                    pantheon_heartbeat.close()
                wandb.log(
                    {
                        "train/global_step": latest_mfu_snapshot.active_step,
                        **latest_mfu_snapshot.history(),
                    },
                    step=latest_mfu_snapshot.active_step,
                )
    elif unroll_decode_eval_only_mode:
        logger.info("Launching unroll-decode evals only mode")
        checkpoint = pref_tag + f"latest.{latest_format}"
        launch_unroll_decode_eval(
            rank,
            start_epoch,
            folder,
            checkpoint,
            cfgs_unroll_decode_evals,
            cfgs_model,
            cfgs_data,
            cfgs_data_aug,
        )
    else:
        logger.info("Skipping training loop due to plan_only_eval_mode being enabled.")
        checkpoint = pref_tag + f"latest.{latest_format}"
        evaluation_reference = None
        if checkpoint_manager is not None:
            shared_reference = [None]
            if rank == 0:
                existing_alias = checkpoint_manager.read_alias("latest", verify=True)
                if existing_alias is not None:
                    shared_reference[0] = existing_alias["checkpoint"]
                elif load_path is not None:
                    reference = checkpoint_manager.register_existing(
                        load_path,
                        global_update=0,
                        epoch=None,
                        metadata={
                            "artifact_kind": "released_qualification_checkpoint",
                            "source_path": os.path.realpath(load_path),
                            "wandb_run_id": trainer.wandb_run_id,
                        },
                    )
                    checkpoint_manager.promote("latest", reference)
                    shared_reference[0] = reference.to_dict()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.broadcast_object_list(shared_reference, src=0)
            evaluation_reference = shared_reference[0]
            if evaluation_reference is None:
                raise RuntimeError("Plan-only qualification could not register an immutable checkpoint")
            checkpoint = evaluation_reference["relative_path"]
        launch_planning_evals(
            rank,
            start_epoch,
            folder,
            checkpoint,
            cfgs_plan_evals,
            cfgs_model,
            cfgs_data,
            cfgs_data_aug,
            "-plan-only",
            world_model=world_model,
            dset=val_traj_dataset,
            preprocessor=preprocessor,
            checkpoint_folder=checkpoint_folder,
            checkpoint_reference=evaluation_reference,
            checkpoint_manager=checkpoint_manager,
            wandb_trainer=trainer,
            final_epoch=True,
        )


def launch_planning_evals(
    rank,
    epoch,
    folder,
    checkpoint,
    cfgs_plan_evals,
    cfgs_model,
    cfgs_data,
    cfgs_data_aug,
    tag_suffix,
    world_model=None,
    dset=None,
    preprocessor=None,
    checkpoint_folder=None,
    checkpoint_reference=None,
    checkpoint_manager=None,
    wandb_trainer=None,
    final_epoch=False,
):
    """
    Launch planning evaluations for the current training checkpoint.

    This function generates complete eval configs by merging training model/data settings
    with eval config templates, then either submits distributed eval jobs via sbatch
    or runs them locally (if separate=False).

    The eval config generation flow:
    1. Load eval config templates from cfgs_plan_evals["eval_cfg_paths"]
       (typically located in configs/online_plan_evals/)
    2. Call build_plan_eval_args() to merge training configs (model, data, data_aug)
       with these templates
    3. Either dump configs for debugging or submit/run eval jobs

    Config options in cfgs_plan_evals:
    - dump_eval_configs (bool): If True, dump generated configs to disk and exit early
      without launching evals. The dump directory is automatically derived from
      eval_cfg_paths (e.g., "configs/online_plan_evals/mz/..." -> "configs/dump_online_evals/mz/").
      Output filenames are derived from the template basenames.
    - separate (bool): If True (default), submit eval jobs via sbatch. If False, run
      evals on rank 0 of the current training job.

    To generate eval configs without running training (e.g., for an already-trained model):
    1. Set meta.plan_only_eval_mode: true in your training config
    2. Set evals.dump_eval_configs: true in your training config
    3. Run: python -m app.main --fname <your_config.yaml> --debug
    4. Configs will be saved to configs/dump_online_evals/<env>/ (derived from eval_cfg_paths)

    Args:
        rank: Process rank in distributed training
        epoch: Current training epoch
        folder: Output folder path for the training run
        checkpoint: Checkpoint filename to evaluate
        cfgs_plan_evals: Evaluation configuration dict from training config
        cfgs_model: Model configuration from training config
        cfgs_data: Data configuration from training config
        cfgs_data_aug: Data augmentation configuration from training config
        tag_suffix: Suffix to append to evaluation tags
        world_model: Optional loaded world model (for non-separate eval mode)
        dset: Optional validation dataset (for non-separate eval mode)
        preprocessor: Optional data preprocessor (for non-separate eval mode)
    """
    eval_cfg_paths = cfgs_plan_evals.get("eval_cfg_paths", None)
    if eval_cfg_paths is None:
        return None
    if isinstance(eval_cfg_paths, (str, bytes)):
        raise TypeError("evals.eval_cfg_paths must be a sequence of config paths, not a string")
    eval_cfg_paths = list(eval_cfg_paths)
    if not eval_cfg_paths:
        logger.info("No planning evaluation configs are enabled; skipping planning evaluation")
        return None
    if any(not isinstance(path, str) or not path.strip() for path in eval_cfg_paths):
        raise ValueError("evals.eval_cfg_paths must contain only non-empty config paths")
    eval_nodes = cfgs_plan_evals.get("nodes", None)
    eval_episodes = cfgs_plan_evals.get("eval_episodes", None)
    eval_low_pri = cfgs_plan_evals.get("low_pri", True)
    separate = cfgs_plan_evals.get("separate", True)

    def record_planning_promotion(report):
        if rank != 0 or not report:
            return
        promotion = report.get("promotion")
        if not promotion or not promotion.get("promoted"):
            return
        if wandb_trainer is not None and wandb_trainer.use_wandb:
            winner = promotion["winner"]
            wandb.run.summary["best_planning/checkpoint_id"] = winner["checkpoint"]["id"]
            wandb.run.summary["best_planning/value"] = winner["selection"]["value"]

    override_cfgs_data = cfgs_plan_evals.get("override_cfgs_data", True)
    override_datasets = cfgs_plan_evals.get("override_datasets", True)
    # task_specification
    evals_obs = cfgs_plan_evals.get("obs", None)
    # planner
    evals_alpha = cfgs_plan_evals.get("alpha", None)
    max_episode_steps = cfgs_plan_evals.get("max_episode_steps", None)
    num_act_stepped = cfgs_plan_evals.get("num_act_stepped", None)
    horizon = cfgs_plan_evals.get("horizon", None)
    evals_decode = cfgs_plan_evals.get("decode", None)
    sum_all_diffs = cfgs_plan_evals.get("sum_all_diffs", None)
    goal_H = cfgs_plan_evals.get("goal_H", None)
    num_elites = cfgs_plan_evals.get("num_elites", None)
    if eval_cfg_paths is not None:
        planning_result_dir = (
            os.path.join(checkpoint_folder, "planning_results")
            if checkpoint_reference is not None
            else None
        )
        planning_registry_path = (
            os.path.join(planning_result_dir, "registry.json")
            if planning_result_dir is not None
            else None
        )
        if checkpoint_reference is not None:
            if checkpoint_manager is None:
                raise RuntimeError("strict planning provenance requires a CheckpointManager")
            pre_launch_report = _run_rank0_planning_controller(
                rank,
                lambda: poll_planning_evaluations(
                    planning_registry_path,
                    manager=checkpoint_manager,
                ),
                "pre-launch planning-evaluation poll",
            )
            record_planning_promotion(pre_launch_report)
        immutable_checkpoint_path = (
            os.path.join(checkpoint_folder, checkpoint_reference["relative_path"])
            if checkpoint_reference is not None
            else None
        )
        eval_nodes, eval_tasks_per_node, args_eval, eval_cpus_per_task = build_plan_eval_args(
            app_name="vjepa_wm",
            folder=folder,
            checkpoint=checkpoint,
            eval_cfg_paths=eval_cfg_paths,
            cfgs_model=cfgs_model,
            cfgs_data=cfgs_data,
            cfgs_data_aug=cfgs_data_aug,
            override_cfgs_data=override_cfgs_data,
            override_datasets=override_datasets,
            tag=f"epoch-{epoch}{tag_suffix}",
            evals_decode=evals_decode,
            sum_all_diffs=sum_all_diffs,
            evals_obs=evals_obs,
            evals_alpha=evals_alpha,
            eval_nodes=eval_nodes,
            eval_episodes=eval_episodes,
            max_episode_steps=max_episode_steps,
            num_act_stepped=num_act_stepped,
            horizon=horizon,
            goal_H=goal_H,
            num_elites=num_elites,
            wrapper_kwargs=cfgs_plan_evals.get("wrapper_kwargs", {}),
            checkpoint_folder=checkpoint_folder,
            checkpoint_id=(checkpoint_reference or {}).get("object_id"),
            checkpoint_checksum=(checkpoint_reference or {}).get("sha256"),
            checkpoint_path=immutable_checkpoint_path,
            checkpoint_step=(checkpoint_reference or {}).get("global_update"),
            planning_result_dir=planning_result_dir,
            planning_eval_id=(
                f"{(checkpoint_reference or {}).get('object_id', 'legacy')}-epoch-{epoch}"
                if checkpoint_reference is not None
                else None
            ),
        )

        if checkpoint_reference is not None:
            promotion_eval_index = cfgs_plan_evals.get("promotion_eval_index", 0)
            if promotion_eval_index is not None:
                promotion_eval_index = int(promotion_eval_index)
                if promotion_eval_index < 0 or promotion_eval_index >= len(args_eval):
                    raise ValueError(
                        f"promotion_eval_index={promotion_eval_index} is outside the {len(args_eval)} eval configs"
                    )
            for index, eval_args in enumerate(args_eval):
                eval_args["planning_provenance"]["promotion_eligible"] = index == promotion_eval_index

        # Dump eval configs if in dump_eval_configs mode (useful for generating configs without training)
        dump_eval_configs = cfgs_plan_evals.get("dump_eval_configs", False)
        if dump_eval_configs:
            if rank == 0:
                # Deduce dump directory from eval_cfg_paths
                # e.g., "configs/online_plan_evals/mz/ng/..." -> "configs/dump_online_evals/mz/"
                first_template = eval_cfg_paths[0] if eval_cfg_paths else None
                if first_template and "online_plan_evals" in first_template:
                    # Extract environment name from path (e.g., "mz", "pt", "wall", "droid")
                    parts = first_template.split("online_plan_evals/")
                    if len(parts) > 1:
                        env_part = parts[1].split("/")[0]  # Get first directory after online_plan_evals/
                        dump_dir = f"configs/dump_online_evals/{env_part}"
                    else:
                        dump_dir = "configs/dump_online_evals"
                else:
                    dump_dir = "configs/dump_online_evals"
                os.makedirs(dump_dir, exist_ok=True)
                dumped_paths = []
                for i, cfg in enumerate(args_eval):
                    # Derive output filename from the config's tag field
                    # e.g., "online_gc_zeroshot/wall_L2_ng_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96/epoch-50-plan-only"
                    # -> "wall_L2_ng_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96.yaml"
                    tag = cfg.get("tag", None)
                    if tag:
                        tag_parts = tag.split("/")
                        if len(tag_parts) >= 2:
                            output_name = tag_parts[-2] + ".yaml"
                        else:
                            output_name = tag_parts[0] + ".yaml"
                    else:
                        output_name = f"eval_config_{i}.yaml"
                    output_path = os.path.join(dump_dir, output_name)
                    dump_yaml(cfg, output_path)
                    dumped_paths.append(output_path)
                logger.info(f"Dumped {len(args_eval)} eval configs:\n" + "\n".join(f"  - {p}" for p in dumped_paths))
            import sys

            sys.exit(0)

        if checkpoint_reference is not None:
            provenances = [eval_args["planning_provenance"] for eval_args in args_eval]
            _run_rank0_planning_controller(
                rank,
                lambda: register_planning_evaluations(
                    planning_registry_path,
                    provenances,
                    execution_mode="asynchronous" if separate else "synchronous",
                ),
                "planning-evaluation registration",
            )

        for i, cfg in enumerate(args_eval):
            args_eval[i] = convert_to_dict_recursive(args_eval[i])

        if separate:
            if rank == 0:
                account, partition, qos = slurm_account_partition_and_qos(low_pri=eval_low_pri)
                logger.info(f"Launching online evals with {account=}, {partition=}, {qos=}")
                with submitit.helpers.clean_env():
                    launch_evals(
                        args_for_evals=args_eval,
                        nodes=eval_nodes,
                        tasks_per_node=eval_tasks_per_node,
                        submitit_folder=os.path.join(folder, "submitit-evals"),
                        account=account,
                        partition=partition,
                        qos=qos,
                        cpus_per_task=eval_cpus_per_task,
                        delay_seconds=5,
                        timeout=120,  # to schedule faster, could be insufficient if using old GPUs making eval slow
                    )
                logger.info(f"Launched online evals from templates {eval_cfg_paths}")
            if checkpoint_reference is not None:
                _run_rank0_planning_controller(
                    rank,
                    lambda: mark_planning_evaluations_launched(
                        planning_registry_path,
                        checkpoint_reference["object_id"],
                    ),
                    "planning-evaluation launch commit",
                )
        else:
            from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import EncPredWM
            from evals.simu_env_planning.eval import main_distributed_episodes_eval as gc_main_dist

            if checkpoint_reference is not None:
                _run_rank0_planning_controller(
                    rank,
                    lambda: mark_planning_evaluations_launched(
                        planning_registry_path,
                        checkpoint_reference["object_id"],
                    ),
                    "synchronous planning-evaluation launch commit",
                )
            planning_process_group = None
            planning_rank_is_active = True
            inprocess_world_size = cfgs_plan_evals.get("inprocess_world_size")
            if inprocess_world_size is not None:
                if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                    raise RuntimeError("evals.inprocess_world_size requires initialized distributed training")
                inprocess_world_size = int(inprocess_world_size)
                training_world_size = torch.distributed.get_world_size()
                if inprocess_world_size <= 0 or inprocess_world_size > training_world_size:
                    raise ValueError(
                        "evals.inprocess_world_size must be in [1, training world size]: "
                        f"got {inprocess_world_size} for world size {training_world_size}"
                    )
                # Every training rank must create groups in identical order.
                # The prefix maps ranks 0..7 to node zero under torchrun, which
                # exactly matches the released DROID planning topology.
                planning_process_group = torch.distributed.new_group(
                    ranks=list(range(inprocess_world_size))
                )
                planning_rank_is_active = rank < inprocess_world_size

            training_rng_state = capture_rng_state(rank=rank)
            try:
                if planning_rank_is_active:
                    world_model.eval()
                    for i, cfg in tqdm(enumerate(args_eval)):
                        eval_tag = cfg.get("tag", None)
                        pretrain_folder = cfg.get("folder", None)
                        folder = os.path.join(pretrain_folder, "simu_env_planning/")
                        if eval_tag is not None:
                            folder = os.path.join(folder, eval_tag)
                        cfg["frameskip"] = cfg["model_kwargs"]["data"]["custom"]["frameskip"]
                        cfg["work_dir"] = folder
                        model = EncPredWM(
                            world_model,
                            action_dim=world_model.action_dim,
                            preprocessor=preprocessor,
                            ctxt_window=cfg["model_kwargs"]["wrapper_kwargs"]["ctxt_window"],
                        )
                        planning_metrics = gc_main_dist(
                            cfg,
                            model=model,
                            dset=dset,
                            preprocessor=preprocessor,
                            rank=rank,
                            device=str(model.device),
                            process_group=planning_process_group,
                        )
                        if (
                            rank == 0
                            and wandb_trainer is not None
                            and wandb_trainer.use_wandb
                            and planning_metrics
                        ):
                            provenance = cfg.get("planning_provenance", {})
                            config_digest = str(provenance.get("eval_config_sha256", f"index-{i}"))
                            metric_prefix = f"planning/{config_digest[:16]}"
                            wandb_step = int(
                                (checkpoint_reference or {}).get("global_update")
                                if (checkpoint_reference or {}).get("global_update") is not None
                                else epoch
                            )
                            scalar_metrics = {
                                f"{metric_prefix}/{key}": float(value)
                                for key, value in planning_metrics.items()
                                if isinstance(value, (int, float, np.number)) and not isinstance(value, bool)
                            }
                            wandb.log(
                                {
                                    "train/global_step": wandb_step,
                                    f"{metric_prefix}/complete": 1,
                                    **scalar_metrics,
                                },
                                step=wandb_step,
                            )
                            wandb.run.summary[f"{metric_prefix}/checkpoint_id"] = (
                                checkpoint_reference or {}
                            ).get("object_id")
                if planning_process_group is not None:
                    # Non-planning ranks wait here; planning collectives above
                    # use only the 8-rank subgroup and cannot strand them.
                    torch.distributed.barrier()
            finally:
                # Planning reseeds Python/NumPy/torch and consumes CUDA RNG.
                # Restore each training rank independently so an uninterrupted
                # next epoch matches continuation from the just-saved state.
                restore_rng_state(training_rng_state, strict_cuda=True)
                if planning_process_group is not None and planning_rank_is_active:
                    torch.distributed.destroy_process_group(planning_process_group)

        if checkpoint_reference is not None:
            final_drain_timeout = float(cfgs_plan_evals.get("final_drain_timeout_seconds", 4 * 60 * 60))
            final_drain_interval = float(cfgs_plan_evals.get("final_drain_poll_interval_seconds", 10.0))
            final_drain_required = bool(cfgs_plan_evals.get("final_drain_required", True))
            if final_epoch:
                if separate and final_drain_required and final_drain_timeout <= 0:
                    raise ValueError(
                        "asynchronous final planning drain requires final_drain_timeout_seconds > 0"
                    )
                controller_report = _run_rank0_planning_controller(
                    rank,
                    lambda: drain_planning_evaluations(
                        planning_registry_path,
                        manager=checkpoint_manager,
                        timeout_seconds=final_drain_timeout,
                        poll_interval_seconds=final_drain_interval,
                        raise_on_timeout=final_drain_required,
                    ),
                    "final planning-evaluation drain",
                )
            else:
                controller_report = _run_rank0_planning_controller(
                    rank,
                    lambda: poll_planning_evaluations(
                        planning_registry_path,
                        manager=checkpoint_manager,
                    ),
                    "post-launch planning-evaluation poll",
                )
            record_planning_promotion(controller_report)


def launch_unroll_decode_eval(
    rank,
    epoch,
    folder,
    checkpoint,
    cfgs_unroll_decode_evals,
    cfgs_model,
    cfgs_data,
    cfgs_data_aug,
):
    """
    Launch unroll decode evaluations for counterfactual decoding of unrolled predictions.

    This evaluation hardcodes custom actions (e.g., open/close gripper + move up) to generate
    counterfactual decodings, allowing visual comparison of different action scenarios.

    Config options in cfgs_unroll_decode_evals:
    - dump_eval_configs (bool): If True, dump generated configs to disk and exit early
      without launching evals (useful for generating configs without training)
    - specific_video (bool): If True, use a specific video file instead of dataset samples
    - specific_video_path (str): Path to specific video file (npz format)
    - play_in_reverse (bool): If True, reverse the video sequence
    - obs (str): Observation type - "rgb" or "rgb_state"
    - save_decoding_only (bool): If True, only save decoded predictions (not ground truth comparison)
    - repeat_hardcode_act (int): Number of times to repeat the hardcoded action sequence
    - wrapper_kwargs (dict): Model wrapper configuration (same as evals.wrapper_kwargs)
        - ctxt_window (int): Context window size for the model wrapper
        - proprio_mode (str): Proprioception mode (e.g., "compute_new_pose")

    Args:
        rank: Process rank in distributed training
        epoch: Current training epoch
        folder: Output folder path for the training run
        checkpoint: Checkpoint filename to evaluate
        cfgs_unroll_decode_evals: Unroll decode evaluation configuration dict
        cfgs_model: Model configuration from training config
        cfgs_data: Data configuration from training config
        cfgs_data_aug: Data augmentation configuration from training config
    """
    # Build evaluation arguments
    args_eval = build_unroll_decode_eval_args(
        app_name="vjepa_wm",
        folder=folder,
        checkpoint=checkpoint,
        cfgs_model=cfgs_model,
        cfgs_data=cfgs_data,
        cfgs_data_aug=cfgs_data_aug,
        cfgs_unroll_decode_evals=cfgs_unroll_decode_evals,
        tag=f"epoch-{epoch}",
    )

    # Dump eval configs if in dump_eval_configs mode (useful for generating configs without training)
    dump_eval_configs = cfgs_unroll_decode_evals.get("dump_eval_configs", False)
    if dump_eval_configs:
        if rank == 0:
            dump_dir = "configs/dump_online_evals/vjepa_wm/unroll_decode"
            os.makedirs(dump_dir, exist_ok=True)
            for i, cfg in enumerate(args_eval):
                yaml_path = os.path.join(dump_dir, f"unroll_decode_{i}.yaml")
                dump_yaml(cfg, yaml_path)
            logger.info(f"Dumped {len(args_eval)} unroll_decode eval configs to {dump_dir}")
        # All ranks exit early after dumping configs (skip launching evals)
        return

    for i, cfg in enumerate(args_eval):
        args_eval[i] = convert_to_dict_recursive(args_eval[i])

    # Run eval directly on rank 0
    from evals.unroll_decode.eval import main as unroll_decode_main

    if rank == 0:
        for cfg in args_eval:
            unroll_decode_main(cfg)
