# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from responses_api_agents.ember_bridge_agent.episode import Episode, episode_configuration, validate_result
from responses_api_agents.ember_bridge_agent.source import PreparedBridge


# A separate owner CLI writes real files and handles signals. No simulator or model is faked as a passing task.
CLI = """
import argparse,json,signal,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('command'); p.add_argument('run',nargs='?'); p.add_argument('--config'); p.add_argument('--output')
a=p.parse_args()
if a.command=='validate': raise SystemExit(0)
if a.command=='result': print((Path(a.run)/'evaluation-result.json').read_text()); raise SystemExit(0)
c=json.loads(Path(a.config).read_text()); out=Path(a.output); out.mkdir()
def finish(*args):
    for name,value in {
        'result.json':{'integration':'libero','sdk_summary':{'closed':c.get('mode')!='unclean','faults':[]}},
        'actor/ttc-runtime.json':{'state':'closed'},
        'outcome.json':{'schema':'ember_bridge.outcome.v1','class':'scored','readback_verified':True},
        'evaluation-result.json':{'schema':'ember_bridge.evaluation_result.v1','class':'scored','counted':True,'reward':1}
    }.items():
        path=out/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value))
    raise SystemExit(0)
signal.signal(signal.SIGTERM,finish)
(out/'launched').touch()
if c.get('mode') in ('sleep','unclean'):
    while True: time.sleep(.01)
finish()
"""
SUPERVISOR = """
import argparse,json,signal,subprocess
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--cleanup-receipt'); p.add_argument('--cleanup-id')
p.add_argument('command',nargs=argparse.REMAINDER); a=p.parse_args()
child=subprocess.Popen(a.command[1:])
def stop(*args): child.terminate()
signal.signal(signal.SIGTERM,stop)
status=child.wait()
Path(a.cleanup_receipt).write_text(json.dumps({'identity':a.cleanup_id,'complete':True}))
raise SystemExit(status)
"""


class GradeTest(unittest.TestCase):
    def test_only_verified_binary_native_grades_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            native = Path(temporary)
            outcome = {"schema": "ember_bridge.outcome.v1", "class": "scored", "readback_verified": True}
            (native / "outcome.json").write_text(json.dumps(outcome))
            result = {"schema": "ember_bridge.evaluation_result.v1", "class": "scored", "counted": True}
            for reward in (0, 1, 0.0, 1.0):
                self.assertEqual(validate_result({**result, "reward": reward}, native)["reward"], reward)
            for reward in (True, False, None, float("nan"), float("inf"), -1, 0.5, "1"):
                with self.subTest(reward=reward), self.assertRaises(ValueError):
                    validate_result({**result, "reward": reward}, native)
            outcome["readback_verified"] = False
            (native / "outcome.json").write_text(json.dumps(outcome))
            with self.assertRaises(ValueError):
                validate_result({**result, "reward": 1}, native)

    def test_policy_failure_counts_but_infrastructure_never_counts(self) -> None:
        for classification in (
            "policy_failure.deadline",
            "policy_failure.decision_budget",
            "policy_failure.context",
            "policy_failure.no_progress",
        ):
            result = {"schema": "ember_bridge.evaluation_result.v1", "class": classification, "counted": True}
            self.assertIs(validate_result(result, Path("unused")), result)
        for classification in ("infrastructure", "provider_blocked", "invalid", "missing", "foreign"):
            result = {"schema": "ember_bridge.evaluation_result.v1", "class": classification, "counted": False}
            self.assertIs(validate_result(result, Path("unused")), result)
            with self.assertRaises(ValueError):
                validate_result({**result, "counted": True}, Path("unused"))


class LifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        files = {
            "ember_bridge/__init__.py": "",
            "ember_bridge/cli.py": CLI,
            "nanoverse/__init__.py": "",
            "nanoverse/environment/__init__.py": "",
            "nanoverse/environment/supervise.py": SUPERVISOR,
            "nanoverse/environment/paths.py": "def revision(): return 'test-revision'\n",
            "nanoverse/environment/processes.py": (
                "import json\nfrom pathlib import Path\n"
                "def cleanup_receipt_valid(path,identity):\n"
                "    v=json.loads(Path(path).read_text()); return v=={'identity':identity,'complete':True}\n"
            ),
        }
        for name, data in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(data)
        self.output = self.root / "output"
        self.output.mkdir()
        self.runtime = PreparedBridge(
            self.root, Path(sys.executable), self.root / "wheel", "node", {"nanoverse_revision": "test-revision"}
        )

    def episode(self, mode: str, *, timeout: float = 5) -> Episode:
        return Episode(
            self.runtime,
            self.output,
            {"mode": mode, "native": {"python": sys.executable}},
            {"PATH": os.defpath},
            timeout,
            5,
        )

    async def launched(self) -> None:
        async with asyncio.timeout(5):
            while not (self.output / "native/launched").exists():
                await asyncio.sleep(0.01)

    async def run_after_launch(self, episode: Episode) -> dict[str, object]:
        """Start a short test deadline only after the fixture installs its signal handler."""
        original = asyncio.create_subprocess_exec

        async def ready(*args: str, **kwargs: object) -> asyncio.subprocess.Process:
            process = await original(*args, **kwargs)
            if any(arg.endswith("/launch.py") for arg in args):
                await self.launched()
            return process

        with patch("responses_api_agents.ember_bridge_agent.episode.asyncio.create_subprocess_exec", ready):
            return await episode.run()

    async def test_completed_episode_reaps_supervisor_and_checks_receipts(self) -> None:
        episode = self.episode("success")
        self.assertEqual((await episode.run())["reward"], 1)
        self.assertTrue(episode.cleanup_confirmed)
        self.assertIsNotNone(episode.process.returncode)

    async def test_deadline_waits_for_owner_cleanup(self) -> None:
        episode = self.episode("sleep", timeout=0.3)
        with self.assertRaises(TimeoutError):
            await self.run_after_launch(episode)
        self.assertTrue(episode.cleanup_confirmed)
        self.assertIsNotNone(episode.process.returncode)

    async def test_cancellation_waits_for_owner_cleanup(self) -> None:
        episode = self.episode("sleep")
        task = asyncio.create_task(episode.run())
        await self.launched()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(episode.cleanup_confirmed)
        self.assertIsNotNone(episode.process.returncode)

    async def test_cancellation_during_spawn_still_reaps_owned_process(self) -> None:
        episode = self.episode("success")
        original = asyncio.create_subprocess_exec
        spawning = asyncio.Event()

        async def delayed(*args: str, **kwargs: object) -> asyncio.subprocess.Process:
            if any(arg.endswith("/supervise.py") for arg in args):
                spawning.set()
                await asyncio.sleep(0.1)
                process = await original(*args, **kwargs)
                await self.launched()
                return process
            return await original(*args, **kwargs)

        with patch("responses_api_agents.ember_bridge_agent.episode.asyncio.create_subprocess_exec", delayed):
            task = asyncio.create_task(episode.run())
            await spawning.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertIsNotNone(episode.process.returncode)
        # A supervisor killed before its handler starts may have no receipt: never claim cleanup in that case.

    async def test_unconfirmed_sdk_shutdown_is_rejected(self) -> None:
        episode = self.episode("unclean", timeout=0.3)
        with self.assertRaisesRegex(RuntimeError, "SDK/TTC"):
            await self.run_after_launch(episode)
        self.assertFalse(episode.cleanup_confirmed)

    async def test_abrupt_worker_death_signals_the_owner_supervisor(self) -> None:
        launch = Path(__file__).parents[1] / "launch.py"
        config = self.output / "config.json"
        config.write_text(json.dumps({"mode": "sleep"}))
        arguments = [
            str(launch),
            "PARENT",
            str(self.root / "nanoverse/environment/supervise.py"),
            "--cleanup-receipt",
            str(self.output / "cleanup.json"),
            "--cleanup-id",
            "parent-death",
            "--",
            sys.executable,
            "-m",
            "ember_bridge.cli",
            "run",
            "--config",
            str(config),
            "--output",
            str(self.output / "native"),
        ]
        parent_code = (
            "import json,os,subprocess,sys,time; args=json.loads(sys.argv[1]); "
            "args[1]=str(os.getpid()); "
            "p=subprocess.Popen([sys.executable,*args],start_new_session=True); "
            "print(p.pid,flush=True); time.sleep(60)"
        )
        parent = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            parent_code,
            json.dumps(arguments),
            cwd=self.root,
            env={"PATH": os.defpath},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        supervisor_pid = int(await parent.stdout.readline())
        try:
            await self.launched()
            parent.kill()
            await parent.wait()
            async with asyncio.timeout(5):
                while not (self.output / "cleanup.json").exists():
                    await asyncio.sleep(0.01)
            receipt = json.loads((self.output / "cleanup.json").read_text())
            self.assertEqual(receipt, {"identity": "parent-death", "complete": True})
            self.assertTrue(json.loads((self.output / "native/result.json").read_text())["sdk_summary"]["closed"])
        finally:
            if parent.returncode is None:
                parent.kill()
            await parent.wait()
            try:
                os.killpg(supervisor_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


class ConfigurationTest(unittest.TestCase):
    def test_binding_keeps_template_and_rejects_auxiliary_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "harnesses").mkdir()
            (root / "harnesses/codex.mts").touch()
            (root / "harnesses/registry.json").write_text(
                json.dumps({"codex": {"entry": "harnesses/codex.mts", "executable_option": "--codex"}})
            )
            runtime = PreparedBridge(root, root / "python", root / "wheel", "node", {})
            template = {
                "integration": "libero",
                "roles": {"actor": {}},
                "policies": [],
                "model": {},
                "sdk": {"execution": "ttc", "python": "/opt/sdk/bin/python3"},
                "native": {"max_steps": 123},
            }

            def bind() -> dict[str, object]:
                return episode_configuration(
                    template,
                    runtime,
                    harness="codex",
                    executable="codex",
                    cell="libero_goal/put_the_bowl_on_the_plate",
                    init_index=2,
                    seed=7,
                    model_url="http://127.0.0.1:1234/v1",
                )

            bound = bind()
            self.assertEqual(template["native"], {"max_steps": 123})
            self.assertEqual(bound["native"]["max_steps"], 123)
            self.assertEqual(bound["native"]["init_index"], 2)
            self.assertEqual(bound["sdk"]["python"], "/opt/sdk/bin/python3")
            template["roles"]["critic"] = {}
            with self.assertRaises(ValueError):
                bind()
            del template["roles"]["critic"]
            template["sdk"]["execution"] = "trusted_host"
            with self.assertRaises(ValueError):
                bind()
