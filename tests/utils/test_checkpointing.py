# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json
import random
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.utils.checkpointing import (
    CheckpointIntegrityError,
    CheckpointManager,
    CheckpointSchemaError,
    ResumeContractError,
    atomic_json_dump,
    build_checkpoint_v2,
    build_resume_contract,
    checkpoint_epoch,
    checkpoint_global_update,
    config_hash,
    create_checkpoint_manifest,
    file_integrity,
    get_git_manifest,
    restore_rng_state,
    should_promote,
    training_state_from_checkpoint,
    validate_checkpoint_v2,
    validate_resume_contract,
    verify_file_integrity,
)


class TestAtomicIntegrityHelpers(unittest.TestCase):
    def test_atomic_json_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "state.json"
            atomic_json_dump({"z": 1, "a": [2, 3]}, target)
            self.assertEqual(json.loads(target.read_text()), {"a": [2, 3], "z": 1})
            self.assertFalse(list(target.parent.glob(".*.tmp-*")))

    def test_integrity_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "object.bin"
            target.write_bytes(b"complete checkpoint")
            expected = file_integrity(target)
            self.assertTrue(verify_file_integrity(target, **expected))
            target.write_bytes(b"tampered checkpoint")
            with self.assertRaises(CheckpointIntegrityError):
                verify_file_integrity(target, **expected)


class TestResumeContract(unittest.TestCase):
    def setUp(self):
        self.config = {
            "folder": "/first/output",
            "nodes": 4,
            "logging": {"wandb": {"project": "one"}},
            "meta": {"read_checkpoint": "/first/latest.pth.tar", "dtype": "bfloat16"},
            "data": {"dataset_path": "/first/droid", "seed": 234},
            "model": {"depth": 12, "dropout": 0.0},
        }

    def test_hash_is_order_independent(self):
        self.assertEqual(config_hash({"b": 2, "a": 1}), config_hash({"a": 1, "b": 2}))

    def test_operational_changes_are_excluded(self):
        saved = build_resume_contract(self.config)
        changed = dict(self.config)
        changed.update(folder="/second/output", nodes=8)
        changed["logging"] = {"wandb": {"project": "two"}}
        changed["meta"] = dict(changed["meta"], read_checkpoint="/second/latest.pth.tar")
        changed["data"] = dict(changed["data"], dataset_path="/second/droid")
        validate_resume_contract(saved, changed)

    def test_mathematical_change_is_rejected_with_path(self):
        saved = build_resume_contract(self.config)
        changed = dict(self.config)
        changed["model"] = dict(changed["model"], depth=24)
        with self.assertRaisesRegex(ResumeContractError, "model.depth"):
            validate_resume_contract(saved, changed)

    def test_corrupt_saved_contract_is_rejected(self):
        saved = build_resume_contract(self.config)
        saved["config"]["model"]["depth"] = 99
        with self.assertRaisesRegex(ResumeContractError, "internally inconsistent"):
            validate_resume_contract(saved, self.config)


class TestCheckpointSchema(unittest.TestCase):
    def _manifest(self):
        return create_checkpoint_manifest(
            resolved_config={"model": {"depth": 12}},
            dataset_manifest={"name": "DROID", "manifest_sha256": "a" * 64},
            encoder_manifest={"name": "dinov3", "repo_revision": "commit", "weights_sha256": "b" * 64},
            wandb_run_id="run-123",
            git_manifest={"available": True, "commit": "c" * 40, "dirty_patch": ""},
        )

    def test_v2_schema_and_legacy_accessors(self):
        checkpoint = build_checkpoint_v2(
            training_state={
                "predictor": {"weight": 1},
                "optimizer": {"state": {}},
                "scaler": None,
                "schedulers": {"lr": {"_step": 5}, "weight_decay": {"_step": 5}},
            },
            epoch=2,
            global_update=17,
            rng_state={"world_size": 1, "states": {}},
            sampler_state={"epoch": 2, "consumed_batches": 3},
            manifest=self._manifest(),
            legacy_fields={"predictor": {"weight": 1}, "opt": {"state": {}}},
        )
        self.assertTrue(validate_checkpoint_v2(checkpoint))
        self.assertEqual(checkpoint_epoch(checkpoint), 2)
        self.assertEqual(checkpoint_global_update(checkpoint), 17)
        self.assertIn("optimizer", training_state_from_checkpoint(checkpoint))

        legacy = {"epoch": 4, "global_update": 30, "opt": {"state": {}}, "predictor": {}}
        self.assertEqual(checkpoint_epoch(legacy), 4)
        self.assertEqual(checkpoint_global_update(legacy), 30)
        self.assertEqual(training_state_from_checkpoint(legacy)["optimizer"], legacy["opt"])

    def test_missing_scheduler_state_is_rejected(self):
        with self.assertRaisesRegex(CheckpointSchemaError, "schedulers"):
            build_checkpoint_v2(
                training_state={"predictor": {}, "optimizer": {}, "scaler": None},
                epoch=0,
                global_update=0,
                rng_state={},
                sampler_state={},
                manifest=self._manifest(),
            )

    def test_manifest_records_lineage_and_requires_complete_git_when_requested(self):
        empty_patch_sha = hashlib.sha256(b"").hexdigest()
        lineage = [{"relation": "continued_pretraining_fork", "checkpoint_sha256": "d" * 64}]
        manifest = create_checkpoint_manifest(
            resolved_config={"model": {"depth": 12}},
            dataset_manifest={"name": "DROID"},
            encoder_manifest={"name": "dinov3"},
            wandb_run_id="child-run",
            lineage=lineage,
            git_manifest={
                "available": True,
                "commit": "c" * 40,
                "dirty": False,
                "dirty_patch": "",
                "dirty_patch_sha256": empty_patch_sha,
                "status_porcelain": "",
            },
            require_git_manifest=True,
        )
        self.assertEqual(manifest["lineage"], lineage)
        with self.assertRaisesRegex(ValueError, "Git source manifest"):
            create_checkpoint_manifest(
                resolved_config={},
                dataset_manifest={"name": "DROID"},
                encoder_manifest={"name": "dinov3"},
                wandb_run_id="run",
                git_manifest={"available": False, "error": "no .git"},
                require_git_manifest=True,
            )


class TestCheckpointManager(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.manager = CheckpointManager(self.root, prefix="test")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _reference(self, step, contents=None, pending=False):
        source = self.root / f"source-{step}.pth.tar"
        source.write_bytes(contents or f"checkpoint-{step}".encode())
        return self.manager.register_existing(source, global_update=step, epoch=step // 10, pending=pending)

    def test_latest_is_monotonic_without_force(self):
        newer = self._reference(20)
        older = self._reference(10)
        self.assertTrue(self.manager.promote("latest", newer))
        self.assertFalse(self.manager.promote("latest", older))
        self.assertEqual(self.manager.resolve("latest"), self.manager.verify(newer))

    def test_best_metric_direction_and_tie_break(self):
        first = self._reference(10)
        worse = self._reference(20)
        tied_newer = self._reference(30)
        self.assertTrue(
            self.manager.promote(
                "best_rollout", first, metric_name="val/loss", metric_value=0.2, mode="min"
            )
        )
        self.assertFalse(
            self.manager.promote(
                "best_rollout", worse, metric_name="val/loss", metric_value=0.3, mode="min"
            )
        )
        self.assertTrue(
            self.manager.promote(
                "best_rollout", tied_newer, metric_name="val/loss", metric_value=0.2, mode="min"
            )
        )
        alias = self.manager.read_alias("best_rollout")
        self.assertEqual(alias["checkpoint"]["object_id"], tied_newer.object_id)

    def test_restart_repairs_rollout_role_from_latest_embedded_metrics(self):
        candidate = self._reference(20)
        candidate_dict = candidate.to_dict()
        candidate_dict["metadata"]["promotion_metrics"] = {
            "best_rollout": {
                "metric_name": "val/rollout",
                "metric_value": 0.125,
                "mode": "min",
                "count": 1024,
            }
        }
        self.manager.promote("latest", candidate_dict)
        self.assertIsNone(self.manager.read_alias("best_rollout"))

        report = self.manager.repair_embedded_promotions()

        self.assertTrue(report["repaired"]["best_rollout"]["promoted"])
        repaired = self.manager.read_alias("best_rollout", verify=True)
        self.assertEqual(repaired["checkpoint"]["object_id"], candidate.object_id)
        self.assertEqual(repaired["promotion"]["metric_value"], 0.125)

        second_report = self.manager.repair_embedded_promotions()
        self.assertFalse(second_report["repaired"]["best_rollout"]["promoted"])

    def test_alias_verification_fails_after_object_tampering(self):
        reference = self._reference(10)
        self.manager.promote("latest", reference)
        self.manager.verify(reference).write_bytes(b"corrupt")
        with self.assertRaises(CheckpointIntegrityError):
            self.manager.resolve("latest")

    def test_alias_verification_binds_the_exact_promotion_metrics(self):
        reference = self._reference(10)
        self.manager.promote(
            "best_rollout",
            reference,
            metric_name="val/rollout",
            metric_value=0.25,
            mode="min",
            metrics={"h1": 0.2, "h2": 0.3},
        )
        alias = self.manager.read_alias("best_rollout")
        receipt_path = self.manager.root / alias["promotion_receipt"]["relative_path"]
        receipt_path.write_text("{}\n", encoding="utf-8")

        with self.assertRaises(CheckpointIntegrityError):
            self.manager.read_alias("best_rollout", verify=True)

    def test_alias_verification_binds_all_checkpoint_reference_metadata(self):
        reference = self._reference(10)
        self.manager.promote("latest", reference)
        alias_path = self.manager.alias_path("latest")
        alias = json.loads(alias_path.read_text(encoding="utf-8"))
        alias["checkpoint"]["metadata"]["promotion_metrics"] = {
            "best_rollout": {
                "metric_name": "forged",
                "metric_value": -1.0,
                "mode": "min",
            }
        }
        alias_path.write_text(json.dumps(alias), encoding="utf-8")

        with self.assertRaisesRegex(CheckpointSchemaError, "does not match"):
            self.manager.read_alias("latest", verify=True)

    def test_gc_retains_roles_and_pending_candidates(self):
        latest = self._reference(10)
        pending = self._reference(20, pending=True)
        orphan = self._reference(30)
        orphan_path = self.manager.verify(orphan)
        self.manager.promote("latest", latest)
        removed = self.manager.garbage_collect()
        self.assertEqual([path.resolve() for path in removed], [orphan_path.resolve()])
        self.assertTrue(self.manager.verify(latest).exists())
        self.assertTrue(self.manager.verify(pending).exists())
        self.assertFalse(removed[0].exists())

    def test_gc_can_retain_three_recent_fallbacks_in_addition_to_roles(self):
        manager = CheckpointManager(self.root / "recent", prefix="test", keep_recent=3)
        references = []
        for step in (10, 20, 30, 40, 50):
            source = self.root / f"recent-source-{step}.pth.tar"
            source.write_bytes(f"checkpoint-{step}".encode())
            references.append(manager.register_existing(source, global_update=step, epoch=step // 10))
        manager.promote("best_rollout", references[0], metric_name="val/loss", metric_value=0.1, mode="min")

        removed = manager.garbage_collect()

        self.assertEqual([path.name for path in removed], [references[1].object_id])
        retained = {references[0].object_id, references[2].object_id, references[3].object_id, references[4].object_id}
        self.assertEqual({path.name for path in manager.objects_dir.glob("*.pth.tar")}, retained)

    def test_gc_removes_obsolete_promotion_receipts_with_their_object(self):
        first = self._reference(10)
        second = self._reference(20)
        self.manager.promote(
            "best_rollout", first, metric_name="val/loss", metric_value=0.3, mode="min"
        )
        first_alias = self.manager.read_alias("best_rollout")
        first_receipt = self.manager.root / first_alias["promotion_receipt"]["relative_path"]
        self.manager.promote(
            "best_rollout", second, metric_name="val/loss", metric_value=0.2, mode="min"
        )

        self.manager.garbage_collect()

        self.assertFalse(first_receipt.exists())
        self.assertFalse((self.manager.objects_dir / first.object_id).exists())

    def test_reference_cannot_escape_object_store(self):
        reference = self._reference(10)
        malicious = reference.to_dict()
        malicious["relative_path"] = "../outside.pth.tar"
        with self.assertRaises(CheckpointSchemaError):
            self.manager.verify(malicious)


class TestPromotionComparison(unittest.TestCase):
    def test_max_min_and_nan(self):
        self.assertTrue(should_promote(2.0, 1.0, mode="max"))
        self.assertTrue(should_promote(1.0, 2.0, mode="min"))
        self.assertFalse(should_promote(float("nan"), 1.0, mode="max"))

    def test_tie_break(self):
        self.assertTrue(should_promote(1.0, 1.0, mode="max", candidate_step=2, incumbent_step=1))
        self.assertFalse(
            should_promote(1.0, 1.0, mode="max", candidate_step=2, incumbent_step=1, tie_break="keep")
        )


class TestGitManifest(unittest.TestCase):
    def test_captures_commit_and_dirty_patch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", root], check=True)
            subprocess.run(["git", "-C", root, "config", "user.email", "test@example.com"], check=True)
            subprocess.run(["git", "-C", root, "config", "user.name", "Test"], check=True)
            tracked = root / "tracked.txt"
            tracked.write_text("before\n")
            subprocess.run(["git", "-C", root, "add", "tracked.txt"], check=True)
            subprocess.run(["git", "-C", root, "commit", "-qm", "initial"], check=True)
            tracked.write_text("after\n")
            untracked = root / "untracked.txt"
            untracked.write_text("untracked contents\n")
            manifest = get_git_manifest(root)
            self.assertTrue(manifest["available"])
            self.assertTrue(manifest["dirty"])
            self.assertIn("tracked.txt", manifest["dirty_patch"])
            self.assertIn("untracked.txt", manifest["dirty_patch"])
            self.assertIn("untracked contents", manifest["dirty_patch"])
            self.assertTrue(manifest["untracked_files_in_patch"])
            self.assertEqual(len(manifest["commit"]), 40)


class TestRNGState(unittest.TestCase):
    def test_cpu_rng_round_trip(self):
        try:
            import numpy as np
            import torch
            from src.utils.checkpointing import capture_rng_state
        except ImportError:
            self.skipTest("NumPy/PyTorch not installed")

        random.seed(12)
        np.random.seed(12)
        torch.manual_seed(12)
        state = capture_rng_state(rank=0)
        expected = (random.random(), float(np.random.random()), float(torch.rand(1)))
        restore_rng_state(state, strict_cuda=False)
        actual = (random.random(), float(np.random.random()), float(torch.rand(1)))
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
