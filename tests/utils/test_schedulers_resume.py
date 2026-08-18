# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from src.utils.schedulers import CosineWDSchedule, LinearDecaySchedule, WSDSchedule, WarmupCosineSchedule


class FakeOptimizer:
    def __init__(self):
        self.param_groups = [{"lr": 0.0, "weight_decay": 0.0}, {"lr": 0.0, "weight_decay": 0.0}]


class TestSchedulerResume(unittest.TestCase):
    def _assert_exact_resume(self, factory):
        original = factory(FakeOptimizer())
        for _ in range(7):
            original.step()
        state = original.state_dict()

        resumed = factory(FakeOptimizer())
        resumed.load_state_dict(state)
        self.assertEqual(resumed.state_dict(), state)
        for _ in range(5):
            self.assertAlmostEqual(resumed.step(), original.step(), places=14)
        self.assertEqual(resumed.state_dict(), original.state_dict())

    def test_wsd_resume(self):
        self._assert_exact_resume(
            lambda optimizer: WSDSchedule(
                optimizer,
                warmup_steps=3,
                anneal_steps=4,
                T_max=20,
                start_lr=0.001,
                ref_lr=0.01,
                final_lr=0.0001,
            )
        )

    def test_warmup_cosine_resume(self):
        self._assert_exact_resume(
            lambda optimizer: WarmupCosineSchedule(
                optimizer,
                warmup_steps=3,
                start_lr=0.001,
                ref_lr=0.01,
                T_max=20,
                final_lr=0.0001,
            )
        )

    def test_cosine_weight_decay_resume(self):
        self._assert_exact_resume(
            lambda optimizer: CosineWDSchedule(optimizer, ref_wd=0.04, T_max=20, final_wd=0.4)
        )

    def test_linear_decay_resume(self):
        self._assert_exact_resume(
            lambda optimizer: LinearDecaySchedule(optimizer, ref_lr=0.01, max_steps=20, final_lr=0.001)
        )

    def test_legacy_step_only_state_is_accepted(self):
        schedule = LinearDecaySchedule(FakeOptimizer(), ref_lr=0.01, max_steps=20, final_lr=0.001)
        schedule.load_state_dict({"_step": 9})
        self.assertEqual(schedule._step, 9)
        self.assertEqual(schedule.ref_lr, 0.01)

    def test_invalid_state_type_is_rejected(self):
        schedule = LinearDecaySchedule(FakeOptimizer(), ref_lr=0.01, max_steps=20)
        with self.assertRaises(TypeError):
            schedule.load_state_dict([("_step", 1)])


if __name__ == "__main__":
    unittest.main()
