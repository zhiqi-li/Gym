# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import tempfile
import unittest
from pathlib import Path

from responses_api_agents.ember_bridge_agent.attempt import Attempt


class AttemptTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.binding = {"request": {"cell": "task", "init_index": 0}, "source": {"resolved_commit": "a" * 40}}

    def test_saved_reply_replays_without_reserving_another_episode(self) -> None:
        attempt = Attempt.claim(self.root, rollout_id="one", binding=self.binding)
        response = json.dumps({"reward": 1, "evaluation_completed": True})
        attempt.save_response(response)
        attempt.release()
        again = Attempt.claim(self.root, rollout_id="one", binding=self.binding)
        self.assertEqual(again.output, attempt.output)
        self.assertEqual(again.saved_response, response)
        self.assertFalse((self.root / "active.json").exists())
        with self.assertRaisesRegex(RuntimeError, "cannot be replaced"):
            again.save_response("{}")

    def test_interrupted_attempt_never_replays_physical_work(self) -> None:
        attempt = Attempt.claim(self.root, rollout_id="one", binding=self.binding)
        with self.assertRaisesRegex(RuntimeError, "physical replay"):
            Attempt.claim(self.root, rollout_id="one", binding=self.binding)
        with self.assertRaisesRegex(RuntimeError, "cleanup is unconfirmed"):
            Attempt.claim(self.root, rollout_id="another", binding=self.binding)
        attempt.release()
        with self.assertRaisesRegex(RuntimeError, "physical replay"):
            Attempt.claim(self.root, rollout_id="one", binding=self.binding)
        self.assertIsNone(Attempt.claim(self.root, rollout_id="another", binding=self.binding).saved_response)

    def test_cached_reply_is_available_while_another_attempt_is_quarantined(self) -> None:
        completed = Attempt.claim(self.root, rollout_id="completed", binding=self.binding)
        completed.save_response("{}")
        completed.release()
        quarantined = Attempt.claim(self.root, rollout_id="unclean", binding=self.binding)
        replay = Attempt.claim(self.root, rollout_id="completed", binding=self.binding)
        self.assertEqual(replay.saved_response, "{}")
        with self.assertRaisesRegex(RuntimeError, "another attempt"):
            replay.release()
        self.assertEqual(json.loads((self.root / "active.json").read_text())["episode"], str(quarantined.output))

    def test_changed_task_or_source_cannot_reuse_a_rollout_id(self) -> None:
        attempt = Attempt.claim(self.root, rollout_id="one", binding=self.binding)
        attempt.save_response("{}")
        attempt.release()
        for changed in (
            {**self.binding, "request": {"cell": "other"}},
            {**self.binding, "source": {"resolved_commit": "b" * 40}},
        ):
            with self.assertRaisesRegex(ValueError, "different task or configuration"):
                Attempt.claim(self.root, rollout_id="one", binding=changed)

    def test_partial_response_and_path_like_ids_never_bypass_the_journal(self) -> None:
        attempt = Attempt.claim(self.root, rollout_id="../../escape", binding=self.binding)
        self.assertEqual(attempt.output.parent, self.root)
        (attempt.output / "response.json.tmp").write_text('{"partial":')
        attempt.release()
        with self.assertRaisesRegex(RuntimeError, "physical replay"):
            Attempt.claim(self.root, rollout_id="../../escape", binding=self.binding)
