# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch


try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from nemo_gym.client import ServerClient
    from responses_api_agents.ember_bridge_agent.app import (
        EmberBridgeAgent,
        EmberBridgeAgentConfig,
        LiberoRunRequest,
        actor_environment,
    )
    from responses_api_agents.ember_bridge_agent.proxy import ModelProxy
    from responses_api_agents.ember_bridge_agent.source import BridgeSource, PreparedBridge
except ModuleNotFoundError as error:
    if error.name.startswith(("nemo_gym", "responses_api_agents")):
        raise
    raise unittest.SkipTest(f"Install Gym requirements for endpoint tests: {error.name}") from error


class EndpointTest(unittest.TestCase):
    def test_native_and_policy_rewards_masks_and_duplicate_attempts(self) -> None:
        for classification, counted, reward, metadata, capture_id in (
            ("scored", True, 1, {"_ng_rollout_id": "one-attempt"}, "one-attempt"),
            ("scored", True, 0, {"_ng_task_index": 2, "_ng_rollout_index": 3, "_ng_run_id": "run-one"}, "2-3"),
            (
                "policy_failure.deadline",
                True,
                None,
                {"_ng_task_index": 1, "_ng_rollout_index": 2, "_ng_attempt_index": 2},
                "1-2-a2",
            ),
            ("provider_blocked", False, None, {"_ng_rollout_id": "one-attempt"}, "one-attempt"),
        ):
            with (
                self.subTest(classification=classification, reward=reward),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                manifest = root / "ttc.json"
                manifest.write_text("{}")
                runtime = PreparedBridge(
                    root,
                    root / "bin/python",
                    root / "wheel",
                    "node",
                    {"resolved_commit": "a" * 40, "nanoverse_revision": "test"},
                )
                config = EmberBridgeAgentConfig(
                    host="127.0.0.1",
                    port=8000,
                    name="libero_agent",
                    entrypoint="app.py",
                    num_workers=1,
                    model_server={"type": "responses_api_models", "name": "policy_model"},
                    source=BridgeSource("operator-repository", "main", root / "cache"),
                    bridge_config={},
                    ttc_manifest=manifest,
                    guest_programs=manifest,
                    decisions_key_file=manifest,
                    output_root=root,
                )
                agent = EmberBridgeAgent(config=config, server_client=MagicMock(spec=ServerClient))
                agent._runtime = runtime
                attempts = []
                upstreams = []

                class OwnedEpisode:
                    def __init__(self, *args: object) -> None:
                        self.process = object()
                        self.cleanup_confirmed = True

                    async def run(self) -> dict[str, object]:
                        attempts.append(True)
                        return {"class": classification, "counted": counted, "reward": reward}

                @asynccontextmanager
                async def proxy(*args: object, **kwargs: object):
                    upstreams.append(args[0])
                    yield ModelProxy("http://upstream/v1", {}, 5, base_url="http://127.0.0.1:1234/v1")

                app = FastAPI()
                app.post("/run")(agent.run)
                row = {
                    "responses_create_params": {"input": []},
                    "cell": "libero_goal/put_the_bowl_on_the_plate",
                    **metadata,
                }
                with (
                    patch("responses_api_agents.ember_bridge_agent.app.get_server_url", return_value="http://model"),
                    patch.object(EmberBridgeAgent, "_capture_correlation_enabled", return_value=True),
                    patch.object(
                        EmberBridgeAgent, "_model_call_capture_enabled", return_value=classification == "scored"
                    ),
                    patch("responses_api_agents.ember_bridge_agent.app.model_proxy", proxy),
                    patch("responses_api_agents.ember_bridge_agent.app.episode_configuration", return_value={}),
                    patch("responses_api_agents.ember_bridge_agent.app.Episode", OwnedEpisode),
                    TestClient(app) as client,
                ):
                    result = client.post("/run", json=row)
                    self.assertEqual(result.status_code, 200, result.text)
                    value = result.json()
                    self.assertEqual(value["evaluation_completed"], counted)
                    self.assertEqual(value["mask_sample"], not counted)
                    self.assertEqual(value["reward"], reward or 0)
                    self.assertEqual(value["response"]["output"], [])
                    if classification == "scored":
                        self.assertEqual(value["ng_trajectory"]["rollout_id"], capture_id)
                        self.assertEqual(value["ng_trajectory"]["turns"], [])
                        self.assertEqual(value["ng_trajectory"]["invocations"][0]["status"], "unknown")
                    else:
                        self.assertNotIn("ng_trajectory", value)
                    self.assertEqual(client.post("/run", json=row).json(), value)
                    self.assertEqual(attempts, [True])
                    self.assertEqual(upstreams, [f"http://model/ng-rollout/{capture_id}/v1"])
                    self.assertFalse((root / "active.json").exists())
                    if "_ng_run_id" in metadata:
                        second = {**row, "_ng_run_id": "run-two"}
                        next_result = client.post("/run", json=second).json()
                        self.assertNotEqual(next_result["evidence_dir"], value["evidence_dir"])
                        self.assertEqual(client.post("/run", json=second).json(), next_result)
                        self.assertEqual(attempts, [True, True])
                    for invalid in ({"_ng_run_id": {}}, {"_ng_rollout_id": "bad/path"}):
                        before = len(attempts)
                        self.assertEqual(client.post("/run", json={**row, **invalid}).status_code, 422)
                        self.assertEqual(len(attempts), before)

    def test_credential_environment_and_task_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            runtime = PreparedBridge(root, root / "bin/python", root / "wheel", "/opt/node/bin/node", {})
            with patch.dict(os.environ, {"OPENAI_API_KEY": "secret", "GITLAB_TOKEN": "secret"}):
                environment = actor_environment(
                    runtime, root, manifest, manifest, "local-capability", decisions_key_file=manifest
                )
            self.assertNotIn("OPENAI_API_KEY", environment)
            self.assertNotIn("GITLAB_TOKEN", environment)
            self.assertEqual(environment["NV_API_KEY"], "local-capability")
            self.assertEqual(environment["NANOVERSE_GUEST_PROGRAMS"], str(manifest))
            self.assertEqual(environment["OPENAI_DECISION_API_KEY_FILE"], str(manifest))
            self.assertNotIn("OPENAI_DECISION_API_KEY", environment)
            for init_index in (-1, 50):
                with self.assertRaises(ValueError):
                    LiberoRunRequest(
                        responses_create_params={"input": []},
                        cell="libero_goal/put_the_bowl_on_the_plate",
                        init_index=init_index,
                    )
