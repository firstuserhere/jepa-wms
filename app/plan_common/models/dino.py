# Copyright (c) Facebook, Inc. and its affiliates.
# Inspired from https://github.com/gaoyuezhou/dino_wm
# Licensed under the MIT License
import os
import hashlib
import subprocess
import warnings

import torch
import torch.nn as nn

# Suppress xFormers availability warnings from DINOv2
warnings.filterwarnings("ignore", message="xFormers is not available")

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True


def _sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(path):
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class DinoEncoder(nn.Module):
    def __init__(
        self,
        name,
        feature_key,
        causal_enc=False,
        weights_path=None,
        expected_weights_sha256=None,
        repo_path=None,
        expected_repo_revision=None,
    ):
        super().__init__()
        self.name = name
        if self.name.startswith("dinov2"):
            self.base_model = torch.hub.load("facebookresearch/dinov2", name)
        elif self.name.startswith("dinov3"):
            pretrained_ckpt_root = os.environ.get("JEPAWM_OSSCKPT")
            dinov3_path = repo_path or os.path.join(
                os.environ.get("JEPAWM_HOME", os.path.expanduser("~")), "dinov3"
            )
            if not os.path.isdir(dinov3_path):
                raise FileNotFoundError(
                    f"DINOv3 repository not found at {dinov3_path}. Clone and pin it before training."
                )
            self.repo_revision = _git_revision(dinov3_path)
            if expected_repo_revision is not None and self.repo_revision != expected_repo_revision:
                raise RuntimeError(
                    "DINOv3 repository revision mismatch: "
                    f"expected {expected_repo_revision}, found {self.repo_revision or 'unknown'}"
                )
            if "vitl16" in self.name:
                weights_path = weights_path or (
                    f"{pretrained_ckpt_root}/dinov3/{name}_pretrain_lvd1689m-8aa4cbdd.pth"
                )
                if not os.path.isfile(weights_path):
                    raise FileNotFoundError(f"DINOv3 weights not found at {weights_path}")
                self.weights_sha256 = _sha256_file(weights_path)
                if expected_weights_sha256 is not None and self.weights_sha256 != expected_weights_sha256:
                    raise RuntimeError(
                        "DINOv3 weight checksum mismatch: "
                        f"expected {expected_weights_sha256}, found {self.weights_sha256}"
                    )
                filename_hash = os.path.basename(weights_path).rsplit("-", 1)[-1].removesuffix(".pth")
                if len(filename_hash) == 8 and not self.weights_sha256.startswith(filename_hash):
                    raise RuntimeError(
                        f"DINOv3 filename hash {filename_hash} does not match SHA-256 {self.weights_sha256}"
                    )
                self.base_model = torch.hub.load(
                    dinov3_path,
                    name,
                    source="local",
                    weights=weights_path,
                )
            else:
                weights_path = weights_path or f"{pretrained_ckpt_root}/dinov3/{name}_pretrain_lvd1689m.pth"
                if not os.path.isfile(weights_path):
                    raise FileNotFoundError(f"DINOv3 weights not found at {weights_path}")
                self.weights_sha256 = _sha256_file(weights_path)
                if expected_weights_sha256 is not None and self.weights_sha256 != expected_weights_sha256:
                    raise RuntimeError(
                        "DINOv3 weight checksum mismatch: "
                        f"expected {expected_weights_sha256}, found {self.weights_sha256}"
                    )
                self.base_model = torch.hub.load(
                    dinov3_path,
                    name,
                    source="local",
                    weights=weights_path,
                )
            self.weights_path = os.path.realpath(weights_path)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size

    def forward(self, x):
        emb = self.base_model.forward_features(x)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1)  # dummy patch dim
        return emb
