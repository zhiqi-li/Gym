# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import tempfile
import unittest
from pathlib import Path

from nemo_gym.config import ModelServerRef
from nemo_gym.trajectory import TrajectoryRecord
from responses_api_agents.ember_bridge_agent.proxy import ModelExchange, ModelProxy
from responses_api_agents.ember_bridge_agent.trajectory import observed_trajectory


class TrajectoryTest(unittest.TestCase):
    def test_actual_calls_and_owner_completion_keep_tool_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "native/actor").mkdir(parents=True)
            (root / "native/actor/receipt.json").write_text(json.dumps({"finished": True}))
            proxy = ModelProxy("http://model/v1", {}, 10)
            reasoning = {"type": "reasoning", "summary": []}
            tool = {"type": "function_call", "call_id": "tool-1", "name": "step", "arguments": "{}"}
            proxy.exchanges = [
                ModelExchange(
                    {"input": [{"role": "user", "content": "Move the bowl"}]},
                    started_at=42,
                    response={"id": "real-id", "output": [reasoning, tool]},
                    status=200,
                )
            ]
            record = observed_trajectory(
                proxy,
                model_ref=ModelServerRef(type="responses_api_models", name="policy"),
                rollout_id="1-0-a2",
                output=root,
            )
            self.assertEqual(record.invocations[0].status, "completed")
            self.assertEqual(record.turns[0].timestamp, 42)
            self.assertEqual(record.turns[0].answer, [tool])
            self.assertEqual(record.turns[0].reasoning_content, [reasoning])
            self.assertEqual(record.turns[0].model_calls[0].response_id, "real-id")
            self.assertEqual(record.tool_calls, [])
            self.assertEqual([gap.code for gap in record.gaps], ["tool_execution_unavailable"])
            self.assertEqual(TrajectoryRecord.model_validate_json(record.model_dump_json()), record)

    def test_missing_response_is_an_unresolved_observed_attempt(self) -> None:
        proxy = ModelProxy("http://model/v1", {}, 10)
        proxy.exchanges = [ModelExchange({"input": []}, status=429)]
        with tempfile.TemporaryDirectory() as temporary:
            record = observed_trajectory(
                proxy,
                model_ref=ModelServerRef(type="responses_api_models", name="policy"),
                rollout_id="0-0",
                output=Path(temporary),
            )
        self.assertEqual(record.invocations[0].status, "unknown")
        self.assertEqual(len(record.turns), 1)
        self.assertEqual(record.turns[0].model_calls, [])
        self.assertIn("model_exchange_incomplete", [gap.code for gap in record.gaps])

    def test_no_proxy_does_not_invent_a_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            record = observed_trajectory(
                None,
                model_ref=ModelServerRef(type="responses_api_models", name="policy"),
                rollout_id="0-0",
                output=Path(temporary),
            )
        self.assertEqual(record.turns, [])
        self.assertIn("model_exchange_unavailable", [gap.code for gap in record.gaps])
