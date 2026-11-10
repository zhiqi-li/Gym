# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare a paired Bridge/Nanoverse release from an operator-selected Git revision."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit


@dataclass(frozen=True)
class BridgeSource:
    """A Git source and the tools used to build its own release format."""

    repository: str
    ref: str
    cache_dir: Path
    component_dir: str = "groot/vlm/external_deps"
    python: str = "python3.11"
    node: str = "node"

    def __post_init__(self) -> None:
        parsed = urlsplit(self.repository)
        if not self.repository or self.repository.startswith("-"):
            raise ValueError("repository must name a Git repository")
        if parsed.scheme in {"http", "https"} and (
            parsed.username or parsed.password or parsed.query or parsed.fragment
        ):
            raise ValueError("Use Git's credential helper instead of credentials in the repository URL")
        if not self.ref or self.ref.startswith("-") or any(c.isspace() for c in self.ref):
            raise ValueError("ref must be a Git branch, tag or commit")
        component = PurePosixPath(self.component_dir)
        if component.is_absolute() or ".." in component.parts or not component.parts:
            raise ValueError("component_dir must stay inside the source repository")


@dataclass(frozen=True)
class PreparedBridge:
    """An installed release; one server keeps this revision for its lifetime."""

    root: Path
    python: Path
    wheel: Path
    node: str
    provenance: dict[str, object]


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=600,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if result.returncode:
        raise RuntimeError(f"Git {arguments[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def resolve_revision(source: BridgeSource, mirror: Path) -> str:
    """Resolve a floating ref once, or reuse an already fetched immutable commit."""
    if not mirror.exists():
        subprocess.run(["git", "init", "--bare", str(mirror)], check=True, capture_output=True)
    cached = False
    if re.fullmatch(r"[0-9a-fA-F]{40}", source.ref):
        cached = (
            subprocess.run(
                ["git", "-C", str(mirror), "cat-file", "-e", f"{source.ref}^{{commit}}"],
                capture_output=True,
            ).returncode
            == 0
        )
    if cached:
        return _git(mirror, "rev-parse", f"{source.ref}^{{commit}}")
    _git(mirror, "fetch", "--depth=1", "--filter=blob:none", "--", source.repository, source.ref)
    return _git(mirror, "rev-parse", "FETCH_HEAD^{commit}")


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checkout_source(mirror: Path, checkout: Path, revision: str, components: str) -> None:
    """Materialize the component pair without inheriting the cache's bare setting."""
    _git(mirror, "worktree", "add", "--detach", "--no-checkout", str(checkout), revision)
    component = PurePosixPath(components)
    _git(checkout, "sparse-checkout", "set", "--cone", str(component / "ember_bridge"), str(component / "nanoverse"))
    # sparse-checkout enables worktreeConfig; the bare cache's shared core.bare
    # then also applies here unless this linked worktree overrides it.
    _git(checkout, "config", "--worktree", "core.bare", "false")
    _git(checkout, "checkout", "--detach", revision)


def _installed_files(root: Path) -> dict[str, str]:
    excluded = {"node_modules", "__pycache__", ".git"}
    files = {}
    for directory, children, names in os.walk(root):
        children[:] = sorted(name for name in children if name not in excluded and not name.endswith(".egg-info"))
        for name in sorted(names):
            path = Path(directory) / name
            files[str(path.relative_to(root))] = _digest(path)
    return files


def _load_prepared(root: Path, node: str) -> PreparedBridge:
    receipt = json.loads((root / "prepared.json").read_text())
    artifacts = root / "artifacts"
    for name, expected in receipt["artifacts"].items():
        if Path(name).name != name or _digest(artifacts / name) != expected:
            raise ValueError("Prepared Bridge artifact changed; prepare a fresh runtime")
    if _installed_files(root / "runtime/ember_bridge") != receipt["installed_files"]:
        raise ValueError("Installed Bridge source changed; prepare a fresh runtime")
    (wheel,) = artifacts.glob("nanoverse-*.whl")
    provenance = {key: value for key, value in receipt.items() if key != "installed_files"}
    return PreparedBridge(root / "runtime/ember_bridge", root / "venv/bin/python", wheel, node, provenance)


def prepare_bridge(source: BridgeSource) -> PreparedBridge:
    """Use Bridge's release builder, including its matched Python/npm packages.

    The cache is outside source checkouts because Bridge's builder requires that.
    Updating a branch prepares a different runtime instead of modifying a live one.
    No simulator is installed here, and no source patch is applied by Gym.
    """
    cache = source.cache_dir.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    if any((parent / ".git").exists() for parent in (cache, *cache.parents)):
        raise ValueError("Bridge cache_dir must be outside every Git checkout")
    repository_key = hashlib.sha256(source.repository.encode()).hexdigest()[:16]
    repository_root = cache / repository_key
    repository_root.mkdir(exist_ok=True)
    node = shutil.which(source.node)
    python = shutil.which(source.python)
    if node is None or python is None:
        raise FileNotFoundError("Configure an available Node >=24 executable and Bridge Python >=3.11")
    node_version = subprocess.check_output([node, "--version"], text=True).strip()
    python_version = subprocess.check_output([python, "--version"], text=True).strip()
    if tuple(map(int, python_version.split()[1].split(".")[:2])) < (3, 11):
        raise ValueError("EmberBridge requires Python >=3.11")
    if int(node_version.removeprefix("v").split(".")[0]) < 24:
        raise ValueError("EmberBridge requires Node >=24")
    build_identity = json.dumps([source.component_dir, python_version, node_version])
    tools_key = hashlib.sha256(build_identity.encode()).hexdigest()[:12]
    with (repository_root / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        mirror = repository_root / "repository.git"
        revision = resolve_revision(source, mirror)
        build = repository_root / f"{revision}-{tools_key}"
        if (build / "prepared.json").is_file():
            return _load_prepared(build, node)
        if build.exists():
            raise RuntimeError(f"Incomplete Bridge preparation at {build}; inspect it before retrying")
        build.mkdir()
        checkout = build / "source"
        checkout_source(mirror, checkout, revision, source.component_dir)
        components = PurePosixPath(source.component_dir)
        bridge = checkout / components / "ember_bridge"
        builder = bridge / "scripts/prepare.py"
        if not builder.is_file():
            raise FileNotFoundError("Selected source has no EmberBridge scripts/prepare.py")
        subprocess.run([python, "-m", "venv", str(build / "venv")], check=True)
        interpreter = build / "venv/bin/python"
        env = {**os.environ, "PATH": f"{build / 'venv/bin'}:{Path(node).parent}:{os.environ.get('PATH', '')}"}
        with (build / "build.log").open("wb") as log:

            def run(*command: str, cwd: Path = bridge) -> None:
                subprocess.run(command, cwd=cwd, env=env, check=True, stdout=log, stderr=log, timeout=3600)

            run(str(interpreter), str(builder), "--output", str(build / "artifacts"), "--extras", "runtime")
            artifacts = build / "artifacts"
            runtime = build / "runtime"
            runtime.mkdir()
            with tarfile.open(artifacts / "ember-bridge-source.tar.gz") as archive:
                archive.extractall(runtime, filter="data")
            deployed = runtime / "ember_bridge"
            (wheel,) = artifacts.glob("nanoverse-*.whl")
            run(str(interpreter), "-m", "pip", "install", "--no-deps", str(wheel), cwd=deployed)
            run(
                str(interpreter),
                "-m",
                "pip",
                "install",
                "--find-links",
                str(artifacts),
                "-e",
                ".[runtime]",
                cwd=deployed,
            )
            run("npm", "ci", "--ignore-scripts", cwd=deployed)
            run(str(interpreter), "-m", "ember_bridge.cli", "--help", cwd=deployed)
        release = json.loads((artifacts / "release.json").read_text())
        if release["repository_commit"] != revision:
            raise ValueError("Bridge release does not identify the selected repository commit")
        receipt = {
            "schema": "nemo_gym.ember_bridge.source.v1",
            "repository": source.repository,
            "requested_ref": source.ref,
            "resolved_commit": revision,
            "component_dir": source.component_dir,
            "python_version": python_version,
            "node_version": node_version,
            "nanoverse_revision": release["nanoverse_revision"],
            "release": release,
            "installed_files": _installed_files(deployed),
            "artifacts": {path.name: _digest(path) for path in artifacts.iterdir() if path.is_file()},
        }
        temporary = build / "prepared.json.tmp"
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(build / "prepared.json")
        return _load_prepared(build, node)


def main() -> None:
    """Prepare only project-owned code; native LIBERO remains a separate installation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--ref", default="main")
    parser.add_argument("--component-dir", default="groot/vlm/external_deps")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--python", default="python3.11")
    parser.add_argument("--node", default="node")
    args = parser.parse_args()
    prepared = prepare_bridge(
        BridgeSource(args.repository, args.ref, args.cache_dir, args.component_dir, args.python, args.node)
    )
    print(
        json.dumps(
            {
                "root": str(prepared.root),
                "python": str(prepared.python),
                "wheel": str(prepared.wheel),
                "provenance": prepared.provenance,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
