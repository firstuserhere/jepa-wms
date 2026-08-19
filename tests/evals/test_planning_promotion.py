import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.utils.checkpointing import CheckpointManager
from src.utils.planning_promotion import (
    PlanningDrainTimeout,
    PlanningPromotionError,
    PlanningProvenanceError,
    build_complete_planning_result,
    build_planning_provenance,
    drain_planning_evaluations,
    evaluation_config_sha256,
    load_complete_planning_result,
    load_planning_evaluation_registry,
    mark_planning_evaluations_launched,
    poll_planning_evaluations,
    reconcile_planning_results,
    register_planning_evaluations,
    verify_planning_checkpoint,
    write_complete_planning_result,
)


class PlanningPromotionTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _checkpoint(self, step):
        path = self.root / "checkpoint-objects" / f"checkpoint-step-{step:08d}.pth.tar"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"checkpoint at step {step}".encode()
        path.write_bytes(payload)
        return path, hashlib.sha256(payload).hexdigest()

    def _config(self, checkpoint_path, *, horizon=3, tag="epoch-1"):
        return {
            "folder": str(self.root / "run"),
            "checkpoint_folder": str(checkpoint_path.parent),
            "tag": tag,
            "meta": {"seed": 1, "eval_episodes": 2},
            "distributed": {"distribute_multitask_eval": True},
            "model_kwargs": {
                "checkpoint": str(checkpoint_path),
                "pretrain_kwargs": {"encoder": "dinov3"},
            },
            "task_specification": {"task": "droid-base"},
            "planner": {"planner_name": "adam", "horizon": horizon},
        }

    def _result(self, step, end_distance, *, horizon=3):
        checkpoint_path, checkpoint_digest = self._checkpoint(step)
        config = self._config(checkpoint_path, horizon=horizon, tag=f"epoch-{step}")
        provenance = build_planning_provenance(
            config,
            checkpoint_id=f"step-{step}",
            checkpoint_sha256=checkpoint_digest,
            checkpoint_path=checkpoint_path,
            checkpoint_step=step,
            result_dir=self.root / "planning-results",
            expected_tasks=["droid-base"],
            expected_episodes_per_task=2,
            task_name="droid-base",
        )
        metrics = {
            "ep_end_dist+droid-base": end_distance + 10.0,
            "ep_end_dist_xyz+droid-base": end_distance,
            "episode_success+droid-base": 0.0,
            "ep_end_dist": end_distance + 10.0,
            "ep_end_dist_xyz": end_distance,
            "episode_success": 0.0,
        }
        result = build_complete_planning_result(
            provenance,
            metrics=metrics,
            observed_episode_counts={"droid-base": 2},
        )
        write_complete_planning_result(result)
        return result, provenance

    def _managed_eval(self, manager, step, *, horizon=3, promotion_eligible=True, eval_id=None):
        source, _ = self._checkpoint(step)
        reference = manager.register_existing(source, global_update=step, pending=True)
        immutable_path = manager.verify(reference)
        config = self._config(immutable_path, horizon=horizon, tag=f"epoch-{step}")
        provenance = build_planning_provenance(
            config,
            checkpoint_id=reference.object_id,
            checkpoint_sha256=reference.sha256,
            checkpoint_path=immutable_path,
            checkpoint_step=reference.global_update,
            result_dir=self.root / "managed-results",
            expected_tasks=["droid-base"],
            expected_episodes_per_task=2,
            task_name="droid-base",
            eval_id=eval_id,
            promotion_eligible=promotion_eligible,
        )
        return reference, provenance

    @staticmethod
    def _publish(provenance, end_distance):
        result = build_complete_planning_result(
            provenance,
            metrics={
                "ep_end_dist": end_distance + 10.0,
                "ep_end_dist_xyz": end_distance,
                "episode_success": 0.0,
            },
            observed_episode_counts={"droid-base": 2},
        )
        write_complete_planning_result(result)
        return result

    def test_checkpoint_and_config_provenance_are_exact(self):
        checkpoint_path, checkpoint_digest = self._checkpoint(10)
        config = self._config(checkpoint_path)
        provenance = build_planning_provenance(
            config,
            checkpoint_id="step-10",
            checkpoint_sha256=checkpoint_digest,
            checkpoint_path=checkpoint_path,
            result_dir=self.root / "planning-results",
            expected_tasks=["droid-base"],
            expected_episodes_per_task=2,
            task_name="droid-base",
        )

        self.assertEqual(verify_planning_checkpoint(provenance, checkpoint_path), checkpoint_digest)
        moved_run_config = self._config(checkpoint_path, tag="epoch-999")
        moved_run_config["folder"] = "/different/output/folder"
        moved_run_config.update(
            {"work_dir": "/runtime/eval", "frameskip": 5, "tasks": ["droid-base"], "use_fsdp": False}
        )
        self.assertEqual(evaluation_config_sha256(config), evaluation_config_sha256(moved_run_config))

        checkpoint_path.write_bytes(b"tampered")
        with self.assertRaises(PlanningProvenanceError):
            verify_planning_checkpoint(provenance, checkpoint_path)

    def test_droid_result_is_complete_atomic_and_uses_xyz_end_distance(self):
        result, provenance = self._result(1, 0.75)
        loaded = load_complete_planning_result(provenance["result_path"])

        self.assertEqual(loaded["status"], "complete")
        self.assertEqual(
            loaded["selection"],
            {"metric": "ep_end_dist_xyz", "mode": "min", "value": 0.75},
        )
        self.assertEqual(loaded["checkpoint"]["id"], "step-1")
        self.assertEqual(loaded["integrity_sha256"], result["integrity_sha256"])
        self.assertFalse(list(Path(provenance["result_path"]).parent.glob("*.tmp")))

    def test_incomplete_episode_counts_never_publish(self):
        checkpoint_path, checkpoint_digest = self._checkpoint(2)
        config = self._config(checkpoint_path)
        provenance = build_planning_provenance(
            config,
            checkpoint_id="step-2",
            checkpoint_sha256=checkpoint_digest,
            checkpoint_path=checkpoint_path,
            result_dir=self.root / "planning-results",
            expected_tasks=["droid-base"],
            expected_episodes_per_task=2,
            task_name="droid-base",
        )
        with self.assertRaises(PlanningProvenanceError):
            build_complete_planning_result(
                provenance,
                metrics={"ep_end_dist_xyz": 1.0},
                observed_episode_counts={"droid-base": 1},
            )
        self.assertFalse(Path(provenance["result_path"]).exists())

    def test_reconciliation_is_arrival_order_independent_and_idempotent(self):
        worse, _ = self._result(1, 0.9)
        best, _ = self._result(2, 0.4)
        callbacks = []

        def promote(checkpoint_path, role, metrics):
            callbacks.append((checkpoint_path, role, metrics["ep_end_dist_xyz"]))

        # Pass the newer/better result first to ensure filesystem or completion
        # order has no role in selection.
        outcome = reconcile_planning_results(
            [best["evaluation"]["result_path"], worse["evaluation"]["result_path"]],
            promote=promote,
        )
        self.assertTrue(outcome["promoted"])
        self.assertEqual(outcome["winner"]["checkpoint"]["id"], "step-2")
        self.assertEqual(callbacks, [(best["checkpoint"]["path"], "best_planning", 0.4)])

        late_but_worse, _ = self._result(3, 0.8)
        second_outcome = reconcile_planning_results(
            self.root / "planning-results",
            promote=promote,
            alias_path=outcome["alias_path"],
        )
        self.assertFalse(second_outcome["promoted"])
        self.assertEqual(second_outcome["winner"]["checkpoint"]["id"], "step-2")
        self.assertEqual(len(callbacks), 1)
        self.assertNotIn("latest", late_but_worse["checkpoint"]["path"])

    def test_incomparable_eval_configs_require_an_explicit_filter(self):
        first, _ = self._result(4, 0.5, horizon=3)
        second, _ = self._result(5, 0.3, horizon=6)
        with self.assertRaises(PlanningPromotionError):
            reconcile_planning_results(
                [first["evaluation"]["result_path"], second["evaluation"]["result_path"]]
            )

    def test_checkpoint_manager_promotion_uses_its_native_alias(self):
        source, _ = self._checkpoint(20)
        manager = CheckpointManager(self.root / "managed-checkpoints", prefix="test")
        reference = manager.register_existing(source, global_update=20)
        immutable_path = manager.verify(reference)
        config = self._config(immutable_path)
        provenance = build_planning_provenance(
            config,
            checkpoint_id=reference.object_id,
            checkpoint_sha256=reference.sha256,
            checkpoint_path=immutable_path,
            checkpoint_step=reference.global_update,
            result_dir=self.root / "managed-results",
            expected_tasks=["droid-base"],
            expected_episodes_per_task=2,
            task_name="droid-base",
        )
        result = build_complete_planning_result(
            provenance,
            metrics={"ep_end_dist_xyz": 0.25, "episode_success": 0.0},
            observed_episode_counts={"droid-base": 2},
        )
        write_complete_planning_result(result)

        outcome = reconcile_planning_results([provenance["result_path"]], manager=manager)
        self.assertTrue(outcome["promoted"])
        self.assertEqual(manager.resolve("best_planning"), immutable_path)
        self.assertFalse(reconcile_planning_results([provenance["result_path"]], manager=manager)["promoted"])

    def test_two_eval_configs_release_pending_only_after_both_complete(self):
        manager = CheckpointManager(self.root / "multi-managed", prefix="test")
        reference, primary = self._managed_eval(
            manager,
            30,
            horizon=3,
            promotion_eligible=True,
            eval_id="step-30-primary",
        )
        immutable_path = manager.verify(reference)
        secondary_config = self._config(immutable_path, horizon=6, tag="epoch-30")
        secondary = build_planning_provenance(
            secondary_config,
            checkpoint_id=reference.object_id,
            checkpoint_sha256=reference.sha256,
            checkpoint_path=immutable_path,
            checkpoint_step=reference.global_update,
            result_dir=self.root / "managed-results",
            expected_tasks=["droid-base"],
            expected_episodes_per_task=2,
            task_name="droid-base",
            eval_id="step-30-secondary",
            promotion_eligible=False,
        )
        registry_path = self.root / "managed-results" / "registry.json"
        register_planning_evaluations(
            registry_path,
            [primary, secondary],
            execution_mode="asynchronous",
        )
        registry = load_planning_evaluation_registry(registry_path)
        checkpoint_record = registry["checkpoints"][reference.object_id]
        self.assertEqual(checkpoint_record["launch_state"], "registered")
        records = registry["checkpoints"][reference.object_id]["evaluations"]
        self.assertNotEqual(
            records[primary["eval_id"]]["config_sha256"],
            records[secondary["eval_id"]]["config_sha256"],
        )
        mark_planning_evaluations_launched(registry_path, reference.object_id)
        registry = load_planning_evaluation_registry(registry_path)
        self.assertEqual(registry["checkpoints"][reference.object_id]["launch_state"], "launched")

        self._publish(primary, 0.35)
        partial = poll_planning_evaluations(registry_path, manager=manager)
        self.assertEqual(partial["pending_checkpoint_ids"], [reference.object_id])
        pending = json.loads(manager.pending_path.read_text())["candidates"]
        self.assertIn(reference.object_id, pending)
        # A controller retry must preserve both the launch commit and the
        # already-verified result while the second eval is still outstanding.
        register_planning_evaluations(
            registry_path,
            [primary, secondary],
            execution_mode="asynchronous",
        )
        registry = load_planning_evaluation_registry(registry_path)
        checkpoint_record = registry["checkpoints"][reference.object_id]
        self.assertEqual(checkpoint_record["launch_state"], "launched")
        self.assertEqual(checkpoint_record["status"], "pending")
        self.assertEqual(checkpoint_record["evaluations"][primary["eval_id"]]["status"], "complete")
        self.assertEqual(checkpoint_record["evaluations"][secondary["eval_id"]]["status"], "pending")

        self._publish(secondary, 0.9)
        complete = poll_planning_evaluations(registry_path, manager=manager)
        self.assertEqual(complete["pending_checkpoint_ids"], [])
        self.assertIn(reference.object_id, complete["released_checkpoint_ids"])
        pending = json.loads(manager.pending_path.read_text())["candidates"]
        self.assertNotIn(reference.object_id, pending)
        self.assertEqual(manager.resolve("best_planning"), immutable_path)

    def test_final_drain_observes_delayed_result(self):
        manager = CheckpointManager(self.root / "drain-managed", prefix="test")
        reference, provenance = self._managed_eval(manager, 40, eval_id="step-40-primary")
        registry_path = self.root / "managed-results" / "drain-registry.json"
        register_planning_evaluations(registry_path, [provenance])
        fake_time = [0.0]
        published = [False]

        def clock():
            return fake_time[0]

        def sleep(seconds):
            fake_time[0] += seconds
            if not published[0]:
                self._publish(provenance, 0.2)
                published[0] = True

        report = drain_planning_evaluations(
            registry_path,
            manager=manager,
            timeout_seconds=5,
            poll_interval_seconds=1,
            checkpoint_ids=[reference.object_id],
            raise_on_timeout=True,
            _clock=clock,
            _sleep=sleep,
        )
        self.assertTrue(report["drained"])
        self.assertFalse(report["timed_out"])
        self.assertEqual(report["pending_checkpoint_ids"], [])

    def test_final_drain_timeout_keeps_pending_checkpoint_pinned(self):
        manager = CheckpointManager(self.root / "timeout-managed", prefix="test")
        reference, provenance = self._managed_eval(manager, 50, eval_id="step-50-primary")
        registry_path = self.root / "managed-results" / "timeout-registry.json"
        register_planning_evaluations(registry_path, [provenance])
        fake_time = [0.0]

        def clock():
            return fake_time[0]

        def sleep(seconds):
            fake_time[0] += seconds

        report = drain_planning_evaluations(
            registry_path,
            manager=manager,
            timeout_seconds=2,
            poll_interval_seconds=1,
            _clock=clock,
            _sleep=sleep,
        )
        self.assertTrue(report["timed_out"])
        self.assertEqual(report["pending_checkpoint_ids"], [reference.object_id])
        pending = json.loads(manager.pending_path.read_text())["candidates"]
        self.assertIn(reference.object_id, pending)
        with self.assertRaises(PlanningDrainTimeout):
            drain_planning_evaluations(
                registry_path,
                manager=manager,
                timeout_seconds=0,
                raise_on_timeout=True,
                _clock=clock,
                _sleep=sleep,
            )

    def test_corrupt_result_is_rejected(self):
        result, provenance = self._result(6, 0.2)
        path = Path(provenance["result_path"])
        payload = json.loads(path.read_text())
        payload["selection"]["value"] = 100.0
        path.write_text(json.dumps(payload))
        with self.assertRaises(PlanningProvenanceError):
            load_complete_planning_result(path)

    def test_mutable_latest_checkpoint_is_rejected(self):
        checkpoint = self.root / "jepa-latest.pth.tar"
        checkpoint.write_bytes(b"latest")
        config = self._config(checkpoint)
        with self.assertRaises(PlanningProvenanceError):
            build_planning_provenance(
                config,
                checkpoint_id="latest",
                checkpoint_sha256=hashlib.sha256(b"latest").hexdigest(),
                checkpoint_path=checkpoint,
                result_dir=self.root / "planning-results",
                expected_tasks=["droid-base"],
                expected_episodes_per_task=2,
                task_name="droid-base",
            )


if __name__ == "__main__":
    unittest.main()
