# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bridge CLI lifecycle and its environment-owned evaluation result contract."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import IO
from uuid import uuid4

from responses_api_agents.ember_bridge_agent.source import PreparedBridge


@dataclass
class Episode:
    """Own exactly one Bridge supervisor and verify its shutdown before reuse."""

    runtime: PreparedBridge
    output: Path
    configuration: dict[str, object]
    environment: dict[str, str]
    timeout_seconds: float
    cleanup_seconds: float
    process: asyncio.subprocess.Process | None = None
    cleanup_confirmed: bool = False
    _identity: str = ""
    _log: IO[bytes] | None = None

    async def command(self, *arguments: str, interpreter: Path | None = None) -> bytes:
        """Run a bounded, non-acting Bridge command under its own interpreter."""
        spawning = asyncio.create_task(
            asyncio.create_subprocess_exec(
                str(interpreter or self.runtime.python),
                *arguments,
                cwd=self.runtime.root,
                env=self.environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        )
        try:
            process = await asyncio.shield(spawning)
        except asyncio.CancelledError:
            process = await asyncio.shield(spawning)
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        try:
            async with asyncio.timeout(self.cleanup_seconds):
                stdout, stderr = await process.communicate()
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        if process.returncode:
            raise RuntimeError(stderr.decode(errors="replace")[-2000:] or "Bridge command failed")
        return stdout

    async def run(self) -> dict[str, object]:
        """Validate first, run once, then ask Bridge what the episode measured."""
        config = self.output / "config.json"
        config.write_text(json.dumps(self.configuration, indent=2) + "\n")
        config.chmod(0o600)
        await self.command("-m", "ember_bridge.cli", "validate", "--config", str(config))
        revision = await self.command(
            "-c",
            "from nanoverse.environment.paths import revision; print(revision())",
            interpreter=Path(self.configuration["native"]["python"]),
        )
        if revision.decode().strip() != self.runtime.provenance["nanoverse_revision"]:
            raise ValueError("Native LIBERO must install the wheel from this prepared Bridge release")
        supervisor = (
            (
                await self.command(
                    "-c",
                    "from pathlib import Path; import nanoverse.environment; "
                    "print(Path(nanoverse.environment.__file__).with_name('supervise.py'))",
                )
            )
            .decode()
            .strip()
        )
        self._identity = uuid4().hex
        receipt = self.output / "cleanup.json"
        self._log = (self.output / "launcher.log").open("wb")
        spawning = asyncio.create_task(
            asyncio.create_subprocess_exec(
                str(self.runtime.python),
                str(Path(__file__).with_name("launch.py")),
                str(os.getpid()),
                supervisor,
                "--cleanup-receipt",
                str(receipt),
                "--cleanup-id",
                self._identity,
                "--",
                str(self.runtime.python),
                "-m",
                "ember_bridge.cli",
                "run",
                "--config",
                str(config),
                "--output",
                str(self.output / "native"),
                cwd=self.runtime.root,
                env=self.environment,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=self._log,
                stderr=self._log,
                start_new_session=True,
            )
        )
        try:
            try:
                self.process = await asyncio.shield(spawning)
            except asyncio.CancelledError:
                self.process = await asyncio.shield(spawning)
                raise
            async with asyncio.timeout(self.timeout_seconds):
                await self.process.wait()
            await self.close()
            result = json.loads(await self.command("-m", "ember_bridge.cli", "result", str(self.output / "native")))
            return validate_result(result, self.output / "native")
        finally:
            await self.close()

    async def close(self) -> None:
        """Stop the owned process tree and validate the source owner's cleanup receipt."""
        if self.cleanup_confirmed:
            return
        if self.process is None:
            if self._log is not None:
                self._log.close()
            return
        if self.process.returncode is None:
            try:
                self.process.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                async with asyncio.timeout(self.cleanup_seconds):
                    await self.process.wait()
            except TimeoutError:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await self.process.wait()
                raise RuntimeError("Bridge cleanup timed out; this worker must not run another episode") from None
        if self._log is not None:
            self._log.close()
        check = (
            "import json,sys; from nanoverse.environment.processes import cleanup_receipt_valid; "
            "print(json.dumps(cleanup_receipt_valid(sys.argv[1],sys.argv[2])))"
        )
        valid = json.loads(await self.command("-c", check, str(self.output / "cleanup.json"), self._identity))
        if valid is not True:
            raise RuntimeError("Bridge process cleanup receipt is absent or invalid")
        native = self.output / "native"
        sdk = json.loads((native / "result.json").read_text()).get("sdk_summary")
        ttc = json.loads((native / "actor/ttc-runtime.json").read_text())
        if (
            not isinstance(sdk, dict)
            or sdk.get("closed") is not True
            or sdk.get("faults")
            or ttc.get("state") != "closed"
        ):
            raise RuntimeError("SDK/TTC cleanup is unconfirmed")
        if any(ttc.get(key) for key in ("fault", "cleanup_error", "daemon_cleanup_error", "export_error")):
            raise RuntimeError("TTC reported a shutdown failure")
        self.cleanup_confirmed = True


def validate_result(result: object, native: Path) -> dict[str, object]:
    """Accept native grades and declared policy failures, preserving unmeasured outcomes."""
    if not isinstance(result, dict) or result.get("schema") != "ember_bridge.evaluation_result.v1":
        raise ValueError("Bridge returned an unsupported evaluation result")
    classification = result.get("class")
    if classification == "scored":
        outcome = json.loads((native / "outcome.json").read_text())
        reward = result.get("reward")
        if (
            outcome.get("schema") != "ember_bridge.outcome.v1"
            or outcome.get("class") != "scored"
            or outcome.get("readback_verified") is not True
            or result.get("counted") is not True
            or type(reward) not in (int, float)
            or not math.isfinite(reward)
            or reward not in (0, 1)
        ):
            raise ValueError("LIBERO grade has no verified finite binary native reward")
    elif classification in {
        "policy_failure.decision_budget",
        "policy_failure.deadline",
        "policy_failure.context",
        "policy_failure.no_progress",
    }:
        if result.get("counted") is not True:
            raise ValueError("Bridge did not count its declared policy failure")
    elif result.get("counted") is not False:
        raise ValueError("Unrecognized Bridge outcomes must not count as task measurements")
    return result


def episode_configuration(
    template: dict[str, object],
    runtime: PreparedBridge,
    *,
    harness: str,
    executable: str,
    cell: str,
    init_index: int,
    seed: int,
    model_url: str,
) -> dict[str, object]:
    """Bind the source's registered harness and one task without changing its scoring rules."""
    config = copy.deepcopy(template)
    if config.get("integration") != "libero":
        raise ValueError("This initial integration supports LIBERO only")
    if set(config.get("roles", {})) != {"actor"} or config.get("policies"):
        raise ValueError("LIBERO SDK evaluation requires one actor and no auxiliary policies")
    actor = config["roles"]["actor"]
    if any(actor.get(name) for name in ("model", "tools", "skills", "harness_tools")):
        raise ValueError("Actor model overrides and auxiliary model-capable tools are unsupported")
    if (config.get("sdk") or {}).get("execution") != "ttc":
        raise ValueError("The Gym integration requires sdk.execution=ttc")
    registry = json.loads((runtime.root / "harnesses/registry.json").read_text())
    entry = registry[harness]
    script = (runtime.root / entry["entry"]).resolve(strict=True)
    if not script.is_relative_to(runtime.root.resolve()):
        raise ValueError("Registered harness entry escapes the prepared Bridge")
    command = [runtime.node, str(script)]
    if entry.get("executable_option"):
        command.extend((entry["executable_option"], executable))
    config["harness_command"] = command
    config["provider_auth"] = "api_key"
    config["model"]["base_url"] = model_url
    config["native"].update(cell=cell, init_index=init_index, seed=seed)
    return config
