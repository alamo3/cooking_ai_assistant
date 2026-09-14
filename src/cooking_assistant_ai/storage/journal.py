"""An append-only record of what was said and what was done.

Two cooks went badly and both times the only evidence was memory and a session snapshot that
keeps the last forty turns and then overwrites itself. This keeps everything: every prompt,
every reply, every tool call with its arguments and whether it was accepted, every rejection
reason, every timer. One JSONL file per session, written as it happens, so a cook can be read
back afterwards and the answer to "why did it do that" is a fact rather than a guess.

Deliberately not the database: a log that competes with session snapshots for the same file
is a second source of truth, and a cook must never fail because a log write failed.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

log = logging.getLogger(__name__)

DEFAULT_DIR = Path(os.environ.get("COOK_LOG_DIR", "logs"))


def _safe(value: Any, limit: int = 2000) -> Any:
    """Arguments come from a model and can be anything; never let one break the log."""
    try:
        text = json.dumps(value, default=str)
    except Exception:
        text = str(value)
    return json.loads(text) if len(text) <= limit else text[:limit] + "...(truncated)"


class Journal:
    """One file per session. Every write is best effort; failures are logged, never raised."""

    def __init__(self, session_id: str, directory: Optional[Path] = None,
                 started_at: Optional[datetime] = None):
        self.session_id = session_id
        started = started_at or datetime.now()
        self.directory = Path(directory or DEFAULT_DIR)
        self.path = self.directory / f"{started:%Y-%m-%d}-{session_id}.jsonl"
        self._broken = False

    def write(self, kind: str, **fields: Any) -> None:
        if self._broken:
            return
        record = {"at": datetime.now().isoformat(timespec="seconds"), "kind": kind}
        record.update({k: _safe(v) for k, v in fields.items()})
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:  # a full disk must not end the cook
            self._broken = True
            log.warning("journal disabled (%s): %s", self.path, e)

    # -- the things worth knowing afterwards ---------------------------------

    def heard(self, text: str, proactive: bool = False) -> None:
        self.write("system" if proactive else "cook", text=text)

    def said(self, text: str) -> None:
        if text.strip():
            self.write("assistant", text=text)

    def tool(self, name: str, args: Dict[str, Any], ok: bool, message: str = "") -> None:
        self.write("tool", name=name, args=args, ok=ok, message=message)

    def notice(self, text: str, level: str = "info") -> None:
        self.write("notice", text=text, level=level)

    def note(self, kind: str, **fields: Any) -> None:
        self.write(kind, **fields)


# --------------------------------------------------------------------------- reading back

def list_logs(directory: Optional[Path] = None) -> List[Path]:
    d = Path(directory or DEFAULT_DIR)
    return sorted(d.glob("*.jsonl"), reverse=True) if d.exists() else []


def read(path: Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue  # a torn final line after a hard kill


def render(path: Path, tools: bool = True) -> str:
    """The cook as a readable conversation, with the tool calls interleaved."""
    lines = [f"# {Path(path).name}"]
    for rec in read(path):
        when = rec.get("at", "")[11:19]
        kind = rec.get("kind")
        if kind == "cook":
            lines.append(f"\n{when}  COOK: {rec.get('text', '')}")
        elif kind == "system":
            lines.append(f"\n{when}  [system] {rec.get('text', '')}")
        elif kind == "assistant":
            lines.append(f"{when}  SAID: {rec.get('text', '')}")
        elif kind == "tool" and tools:
            mark = "ok " if rec.get("ok") else "REJECTED"
            args = json.dumps(rec.get("args", {}), ensure_ascii=False)
            lines.append(f"{when}    {mark} {rec.get('name')}({args[:160]})")
            if rec.get("message"):
                lines.append(f"{when}       -> {str(rec['message'])[:200]}")
        elif kind == "notice":
            lines.append(f"{when}    ! {rec.get('text', '')}")
        elif kind == "turn":
            lines.append(f"{when}    ({rec.get('tools', 0)} tool calls, {rec.get('seconds', 0)}s)")
    return "\n".join(lines)


def summarize(path: Path) -> Dict[str, Any]:
    """Counts worth seeing at a glance: how chatty, how many rejections, which tools."""
    out: Dict[str, Any] = {"turns": 0, "tool_calls": 0, "rejected": 0, "by_tool": {},
                           "rejections": [], "first": None, "last": None}
    for rec in read(path):
        out["first"] = out["first"] or rec.get("at")
        out["last"] = rec.get("at")
        if rec.get("kind") == "cook":
            out["turns"] += 1
        elif rec.get("kind") == "tool":
            out["tool_calls"] += 1
            name = rec.get("name", "?")
            out["by_tool"][name] = out["by_tool"].get(name, 0) + 1
            if not rec.get("ok"):
                out["rejected"] += 1
                out["rejections"].append(f"{name}: {str(rec.get('message', ''))[:120]}")
    return out
