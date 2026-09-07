"""Restarting the server from the tablet.

A process cannot reliably restart itself: once it exits there is nothing left to serve the
page that would bring it back. So the lifecycle belongs to a supervisor (the loop at the
bottom of start.ps1), and this module only decides *how* the process ends. Exit with
RESTART_EXIT_CODE and the supervisor relaunches; exit 0 and it stops for good.

The other half is knowing when a restart is worth doing at all: the source fingerprint is
taken at startup and compared on demand, so the tablet can show "code changed" rather than
the cook having to guess whether their edit is live.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Agreed with start.ps1. Anything else is a crash, which the supervisor backs off from.
RESTART_EXIT_CODE = 42

_PACKAGE = Path(__file__).resolve().parent.parent
_WATCHED = (("**/*.py", _PACKAGE), ("web/*.js", _PACKAGE), ("web/*.css", _PACKAGE),
            ("web/*.html", _PACKAGE))


def _snapshot() -> Dict[str, float]:
    """path -> mtime. Cheap enough to run on every status poll (a few hundred stats)."""
    out: Dict[str, float] = {}
    for pattern, root in _WATCHED:
        for p in root.glob(pattern):
            if "__pycache__" in p.parts:
                continue
            try:
                out[str(p)] = p.stat().st_mtime
            except OSError:  # deleted between glob and stat
                continue
    return out


@dataclass
class _State:
    started_at: float
    baseline: Dict[str, float]
    restart_requested: bool = False
    stop_server: Optional[object] = None  # set by the CLI; called to end the process


_state = _State(started_at=time.time(), baseline=_snapshot())


def reset(now: Optional[float] = None) -> None:
    """Re-baseline. Called at startup so the fingerprint reflects the code actually loaded."""
    _state.started_at = now or time.time()
    _state.baseline = _snapshot()
    _state.restart_requested = False


def changed_files() -> List[str]:
    """Source files added, removed or edited since this process started."""
    current = _snapshot()
    names = set(current) | set(_state.baseline)
    changed = [n for n in names if current.get(n) != _state.baseline.get(n)]
    return sorted(os.path.relpath(n, _PACKAGE).replace("\\", "/") for n in changed)


def uptime_s() -> float:
    return time.time() - _state.started_at


def restart_requested() -> bool:
    return _state.restart_requested


def register_stopper(fn) -> None:
    """The CLI hands us a way to end the server loop; without one, restart is unavailable."""
    _state.stop_server = fn


def can_restart() -> bool:
    return _state.stop_server is not None


def request_restart() -> None:
    """Mark the exit as a restart and ask the server to shut down gracefully."""
    if _state.stop_server is None:
        raise RuntimeError("no supervisor: start the server with start.ps1 to enable restarts")
    _state.restart_requested = True
    _state.stop_server()
