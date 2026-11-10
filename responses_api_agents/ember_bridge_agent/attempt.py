# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable ownership of one physical attempt, independent of the HTTP request."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class Attempt:
    """A reserved episode directory or the saved response of that same reservation."""

    output: Path
    saved_response: str | None = None

    @classmethod
    def claim(cls, root: Path, *, rollout_id: str | None, binding: dict[str, object]) -> Attempt:
        """Refuse changed bindings, uncertain prior attempts and quarantined workers.

        The server holds its worker file lock and request lock while using this
        journal. A completed attempt can be read even if another attempt has
        quarantined the worker; replaying its receipt does not start a simulator.
        """
        identity = hashlib.sha256((rollout_id or uuid4().hex).encode()).hexdigest()
        output = root / identity
        fingerprint = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        if output.exists():
            if (output / "identity").read_text() != fingerprint:
                raise ValueError("Rollout ID was already bound to a different task or configuration")
            if (output / "response.json").is_file():
                return cls(output, (output / "response.json").read_text())
            raise RuntimeError("Rollout attempt has no saved result; automatic physical replay is forbidden")
        if (root / "active.json").exists():
            raise RuntimeError("Worker cleanup is unconfirmed; inspect output_root/active.json")
        output.mkdir(mode=0o700)
        (output / "identity").write_text(fingerprint)
        with (root / "active.json").open("x") as stream:
            json.dump({"episode": str(output), "source": binding["source"]}, stream)
        return cls(output)

    def release(self) -> None:
        """Release only this attempt's reservation after the caller confirms cleanup."""
        journal = self.output.parent / "active.json"
        if not journal.exists():
            return
        if json.loads(journal.read_text()).get("episode") != str(self.output):
            raise RuntimeError("Cannot release another attempt's cleanup reservation")
        journal.unlink()

    def save_response(self, response: str) -> None:
        """Commit a complete response without altering the cleanup reservation."""
        if self.saved_response is not None:
            raise RuntimeError("A replayed response cannot be replaced")
        temporary = self.output / "response.json.tmp"
        temporary.write_text(response)
        temporary.replace(self.output / "response.json")
