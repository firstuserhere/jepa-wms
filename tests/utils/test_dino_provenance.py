from pathlib import Path

import torch

from app.plan_common.models import dino


class _Backbone:
    num_features = 1024
    patch_size = 16


def test_dinov3_backbone_uses_official_weights_argument_only(tmp_path: Path, monkeypatch):
    repository = tmp_path / "dinov3"
    repository.mkdir()
    weights = tmp_path / "dinov3_vitl16_native.pth"
    weights.write_bytes(b"native weights")
    calls = []

    monkeypatch.setattr(dino, "_git_revision", lambda _path: "a" * 40)

    def fake_hub_load(repo, name, **kwargs):
        calls.append((repo, name, kwargs))
        return _Backbone()

    monkeypatch.setattr(torch.hub, "load", fake_hub_load)

    encoder = dino.DinoEncoder(
        "dinov3_vitl16",
        "x_norm_patchtokens",
        weights_path=str(weights),
        expected_weights_sha256=dino._sha256_file(weights),
        repo_path=str(repository),
        expected_repo_revision="a" * 40,
    )

    assert encoder.emb_dim == 1024
    assert encoder.patch_size == 16
    assert calls == [
        (
            str(repository),
            "dinov3_vitl16",
            {"source": "local", "weights": str(weights)},
        )
    ]
