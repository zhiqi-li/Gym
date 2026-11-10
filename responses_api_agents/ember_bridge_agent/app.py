# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluation-level LIBERO adapter using the source owner's complete episode loop."""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import IO

from fastapi import Body, FastAPI, HTTPException, Request
from pydantic import Field, JsonValue, PrivateAttr, model_validator

from nemo_gym.agents import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.client import get_server_url
from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, maybe_rollout_id_from_run_body
from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyResponse
from nemo_gym.trajectory import TrajectoryRecord
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.ember_bridge_agent.attempt import Attempt
from responses_api_agents.ember_bridge_agent.episode import Episode, episode_configuration
from responses_api_agents.ember_bridge_agent.proxy import ModelProxy, model_proxy
from responses_api_agents.ember_bridge_agent.source import BridgeSource, PreparedBridge, prepare_bridge
from responses_api_agents.ember_bridge_agent.trajectory import observed_trajectory


LOG = logging.getLogger(__name__)


class EmberBridgeAgentConfig(BaseResponsesAPIAgentConfig):
    """Operator-owned code, simulator and TTC deployment; rows select only the task."""

    model_server: ModelServerRef
    source: BridgeSource
    bridge_config: dict[str, JsonValue]
    ttc_manifest: Path
    guest_programs: Path
    decisions_key_file: Path
    output_root: Path
    harness: str = "opencode"
    harness_executable: str = "opencode"
    episode_timeout_seconds: float = Field(default=7500, gt=0, allow_inf_nan=False)
    cleanup_timeout_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    model_timeout_seconds: float = Field(default=900, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def require_one_worker(self) -> EmberBridgeAgentConfig:
        """One worker owns the simulator, its ephemeral endpoint and the cleanup journal."""
        if self.num_workers != 1:
            raise ValueError("EmberBridge requires num_workers=1; use separate deployments for parallel GPUs")
        if self.skip_verification:
            raise ValueError("LIBERO evaluation requires the source owner's native verification")
        return self


class LiberoRunRequest(BaseRunRequest):
    """A native LIBERO cell and initial state, independent of the caller's prompt."""

    cell: str = Field(pattern=r"^libero_(spatial|object|goal|10|90)/[A-Za-z0-9_]+$")
    init_index: int = Field(default=0, ge=0, le=49)
    seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def require_source_prompt(self) -> LiberoRunRequest:
        """Avoid silently discarding caller-provided task wording or tools."""
        if self.responses_create_params.input or self.responses_create_params.tools:
            raise ValueError("LIBERO task wording and tools belong to the source; input and tools must be empty")
        return self


class LiberoRunResponse(BaseVerifyResponse):
    """Native measurement plus the code and cleanup evidence needed to interpret it."""

    cell: str
    init_index: int
    seed: int
    evaluation_completed: bool
    outcome_class: str
    cleanup_confirmed: bool
    evidence_dir: str
    source: dict[str, JsonValue]
    ng_trajectory: TrajectoryRecord | None = Field(default=None, exclude_if=lambda value: value is None)


def actor_environment(
    runtime: PreparedBridge, output: Path, manifest: Path, programs: Path, capability: str, *, decisions_key_file: Path
) -> dict[str, str]:
    """Give the episode a local capability without inheriting provider credentials."""
    home = output / "home"
    home.mkdir(mode=0o700)
    environment = {
        "HOME": str(home),
        "PATH": f"{runtime.python.parent}:{Path(runtime.node).parent}:/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MUJOCO_GL": "egl",
        "NV_API_KEY": capability,
        "NANOVERSE_TTC_MANIFEST": str(manifest.resolve(strict=True)),
        "NANOVERSE_GUEST_PROGRAMS": str(programs.resolve(strict=True)),
        "OPENAI_DECISION_API_KEY_FILE": str(decisions_key_file.resolve(strict=True)),
    }
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "EGL_DEVICE_ID", "LD_LIBRARY_PATH"):
        if name in os.environ:
            environment[name] = os.environ[name]
    return environment


def response_envelope(body: LiberoRunRequest, proxy: ModelProxy | None, identity: str) -> NeMoGymResponse:
    """Use the last observed model response, or an explicit empty control envelope."""
    if proxy is not None and proxy.last_response is not None:
        return NeMoGymResponse.model_validate(proxy.last_response)
    return NeMoGymResponse(
        id=f"ember-bridge-control-{identity}",
        created_at=0,
        object="response",
        model=body.responses_create_params.model or "ember-bridge",
        output=[],
        parallel_tool_calls=False,
        tools=[],
        tool_choice="auto",
        usage=None,
        metadata={"ember_bridge_envelope": "no_model_response_observed"},
    )


class EmberBridgeAgent(SimpleResponsesAPIAgent):
    """Serialize source-owned episodes and refuse reuse after uncertain cleanup."""

    config: EmberBridgeAgentConfig
    _runtime: PreparedBridge | None = PrivateAttr(default=None)
    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
    _worker_lock: IO[str] | None = PrivateAttr(default=None)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        original = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(application: FastAPI) -> AsyncIterator[None]:
            root = self.config.output_root.expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.config.output_root = root
            self._worker_lock = (root / "worker.lock").open("a")
            try:
                fcntl.flock(self._worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if (root / "active.json").exists():
                    raise RuntimeError("Previous episode cleanup is unconfirmed; inspect output_root/active.json")
                for name in ("ttc_manifest", "guest_programs", "decisions_key_file"):
                    if not getattr(self.config, name).is_file():
                        raise FileNotFoundError(f"Configure {name} to a prepared deployment file")
                executable = shutil.which(self.config.harness_executable)
                if executable is None:
                    raise FileNotFoundError("Configure an installed absolute harness_executable")
                self.config.harness_executable = str(Path(executable).resolve())
                self._runtime = await asyncio.to_thread(prepare_bridge, self.config.source)
                async with original(application):
                    yield
            finally:
                self._worker_lock.close()
                self._worker_lock = None

        app.router.lifespan_context = lifespan
        return app

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        """The source-owned harness exposes a complete episode through /run only."""
        raise NotImplementedError("EmberBridge is an external-agent-loop integration; call /run")

    async def run(self, request: Request, body: LiberoRunRequest = Body()) -> LiberoRunResponse:
        """Run a task once; retries with the same rollout ID replay only a saved receipt."""
        async with self._lock:
            if self._runtime is None:
                raise RuntimeError("Bridge runtime has not been prepared")
            runtime = self._runtime
            # Pydantic drops collector indices from this task's typed request.
            # Resolve the shared identity from the original body before both
            # journaling and model forwarding, even when capture is disabled.
            run_body = await request.json()
            try:
                capture_id = maybe_rollout_id_from_run_body(run_body)
            except (TypeError, ValueError):
                raise HTTPException(status_code=422, detail="Invalid rollout correlation metadata") from None
            run_id = run_body.get("_ng_run_id")
            if run_id is not None and not isinstance(run_id, str):
                raise HTTPException(status_code=422, detail="_ng_run_id must be a string")
            body.capture_rollout_id = capture_id
            capture = self._model_call_capture_enabled() and capture_id is not None
            journal_id = f"{run_id}:{capture_id}" if run_id is not None and capture_id is not None else capture_id
            binding = {
                "request": body.model_dump(mode="json"),
                "run_id": run_id,
                "rollout_id": capture_id,
                "config": self.config.model_dump(mode="json"),
                "source": runtime.provenance,
            }
            attempt = Attempt.claim(self.config.output_root, rollout_id=journal_id, binding=binding)
            if attempt.saved_response is not None:
                return LiberoRunResponse.model_validate_json(attempt.saved_response)
            output = attempt.output
            identity = output.name
            episode: Episode | None = None
            proxy: ModelProxy | None = None
            result: dict[str, object] = {"class": "infrastructure", "counted": False}
            try:
                upstream = self.base_url_for_run(get_server_url(self.config.model_server.name), body) + "/v1"
                async with model_proxy(
                    upstream, dict(request.cookies), self.config.model_timeout_seconds, capture=capture
                ) as proxy:
                    configuration = episode_configuration(
                        self.config.bridge_config,
                        runtime,
                        harness=self.config.harness,
                        executable=self.config.harness_executable,
                        cell=body.cell,
                        init_index=body.init_index,
                        seed=body.seed,
                        model_url=proxy.base_url,
                    )
                    environment = actor_environment(
                        runtime,
                        output,
                        self.config.ttc_manifest,
                        self.config.guest_programs,
                        proxy.capability,
                        decisions_key_file=self.config.decisions_key_file,
                    )
                    episode = Episode(
                        runtime,
                        output,
                        configuration,
                        environment,
                        self.config.episode_timeout_seconds,
                        self.config.cleanup_timeout_seconds,
                    )
                    result = await episode.run()
            except Exception as error:
                LOG.error(
                    "EmberBridge episode %s failed (%s); inspect its evidence directory",
                    identity,
                    type(error).__name__,
                )
                (output / "adapter-error.json").write_text(json.dumps({"type": type(error).__name__}))
            finally:
                safe = episode is None or episode.process is None or episode.cleanup_confirmed
                if safe:
                    attempt.release()
            completed = result.get("counted") is True and safe
            classification = str(result["class"]) if safe else "infrastructure.cleanup"
            response = LiberoRunResponse(
                **body.model_dump(),
                response=response_envelope(body, proxy, identity),
                reward=float(result["reward"]) if completed and classification == "scored" else 0.0,
                mask_sample=not completed,
                evaluation_completed=completed,
                outcome_class=classification,
                cleanup_confirmed=episode.cleanup_confirmed if episode else False,
                failure_kind=None if completed else "ember_bridge:infrastructure",
                failure_reason=None if completed else "Episode was not measured; inspect evidence_dir",
                evidence_dir=str(output),
                source=runtime.provenance,
                ng_trajectory=observed_trajectory(
                    proxy, model_ref=self.config.model_server, rollout_id=capture_id, output=output
                )
                if capture
                else None,
            )
            attempt.save_response(response.model_dump_json(indent=2) + "\n")
            return response


if __name__ == "__main__":
    EmberBridgeAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = EmberBridgeAgent.run_webserver()  # noqa: F401
