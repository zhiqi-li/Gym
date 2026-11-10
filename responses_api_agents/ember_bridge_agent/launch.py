# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind the source owner's Linux supervisor to the Gym worker before executing it.

This runs in the child interpreter, avoiding Python preexec_fn in a threaded web
server. Parent loss before the supervisor installs its signal handler starts no
episode; parent loss afterwards asks that supervisor to reap its descendants.
"""

import ctypes
import os
import signal
import sys


def main() -> None:
    """Execute the owner script with SIGTERM on parent death and a parent-race check."""
    parent = int(sys.argv[1])
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError("Cannot bind Bridge supervisor to its Gym worker")
    if os.getppid() != parent:
        raise SystemExit(127)
    os.execv(sys.executable, [sys.executable, *sys.argv[2:]])


if __name__ == "__main__":
    main()
