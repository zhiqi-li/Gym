# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from responses_api_agents.ember_bridge_agent.source import (
    BridgeSource,
    _digest,
    _installed_files,
    _load_prepared,
    checkout_source,
    prepare_bridge,
    resolve_revision,
)


class SourceTest(unittest.TestCase):
    def test_branch_updates_and_cached_commit_works_without_remote(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "upstream"
            subprocess.run(["git", "init", "-b", "main", str(repository)], check=True, capture_output=True)

            def commit(value: str) -> str:
                (repository / "source").write_text(value)
                for arguments in (
                    ["add", "."],
                    ["-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-m", value],
                ):
                    subprocess.run(["git", "-C", str(repository), *arguments], check=True, capture_output=True)
                return subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()

            first = commit("first")
            source = BridgeSource(str(repository), "main", root / "cache")
            mirror = root / "mirror.git"
            self.assertEqual(resolve_revision(source, mirror), first)
            second = commit("second")
            self.assertEqual(resolve_revision(source, mirror), second)
            repository.rename(root / "unavailable")
            pinned = BridgeSource(str(repository), first, source.cache_dir)
            self.assertEqual(resolve_revision(pinned, mirror), first)
            with self.assertRaisesRegex(RuntimeError, "Git fetch failed"):
                resolve_revision(source, mirror)

    def test_sparse_worktree_from_bare_cache_materializes_both_components(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            upstream = root / "upstream"
            subprocess.run(["git", "init", "-b", "main", str(upstream)], check=True, capture_output=True)
            for component in ("ember_bridge", "nanoverse", "unrelated"):
                target = upstream / "deps" / component / "source.py"
                target.parent.mkdir(parents=True)
                target.write_text(component)
            for args in (
                ["add", "."],
                ["-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-m", "component pair"],
            ):
                subprocess.run(["git", "-C", str(upstream), *args], check=True, capture_output=True)
            mirror = root / "mirror.git"
            revision = resolve_revision(BridgeSource(str(upstream), "main", root / "cache"), mirror)
            checkout = root / "checkout"
            checkout_source(mirror, checkout, revision, "deps")
            self.assertEqual((checkout / "deps/ember_bridge/source.py").read_text(), "ember_bridge")
            self.assertEqual((checkout / "deps/nanoverse/source.py").read_text(), "nanoverse")
            self.assertFalse((checkout / "deps/unrelated").exists())
            bare = subprocess.check_output(["git", "-C", str(mirror), "rev-parse", "--is-bare-repository"], text=True)
            self.assertEqual(bare.strip(), "true")

    def test_rejects_embedded_credentials_and_escaping_component(self) -> None:
        for repository, ref, component in (
            ("https://user:secret@example.org/repo.git", "main", "deps"),
            ("--upload-pack=bad", "main", "deps"),
            ("repo", "--all", "deps"),
            ("repo", "main", "../deps"),
            ("repo", "main", "/deps"),
        ):
            with self.subTest(repository=repository, ref=ref, component=component), self.assertRaises(ValueError):
                BridgeSource(repository, ref, Path("/tmp/cache"), component)

    def test_cache_cannot_live_inside_a_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".git").mkdir()
            with self.assertRaisesRegex(ValueError, "outside"):
                prepare_bridge(BridgeSource("repo", "main", root / "cache"))

    def test_detects_changed_artifact_and_changed_installed_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "artifacts/nanoverse-test.whl"
            artifact.parent.mkdir()
            artifact.write_bytes(b"wheel")
            code = root / "runtime/ember_bridge/main.py"
            code.parent.mkdir(parents=True)
            code.write_text("original")
            receipt = {
                "artifacts": {artifact.name: _digest(artifact)},
                "installed_files": _installed_files(code.parent),
            }
            (root / "prepared.json").write_text(json.dumps(receipt))
            self.assertEqual(_load_prepared(root, "node").wheel, artifact)
            code.write_text("changed")
            with self.assertRaisesRegex(ValueError, "source changed"):
                _load_prepared(root, "node")
            code.write_text("original")
            artifact.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                _load_prepared(root, "node")
