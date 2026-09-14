"""Rejections that outlived their turn.

A tool call and its result exist only inside the turn that made them: the next turn sees a
freshly rebuilt state and the last few sentences of conversation, and nothing else. So a
rejected call was genuinely invisible one turn later. The cook watched the assistant carry on
as though a failed call had worked, and it was not ignoring the failure, it could not see it.

Storing it is the exception to everything else here being derived, because a rejection is an
event rather than a fact about the world. It stops being stored the moment the same tool
succeeds.
"""
from __future__ import annotations

from typing import Any, Dict, List

from cooking_assistant_ai.core.fmt import fmt_time
from cooking_assistant_ai.model.types import Session


def render(session: Session) -> str:
    """The block the next turn reads. Empty when nothing is outstanding."""
    if not session.open_failures:
        return ""
    lines = ["UNRESOLVED (these tool calls were rejected and never put right)"]
    for f in session.open_failures:
        lines.append(f"  {f.tool} at {fmt_time(f.at)}: {f.reason}")
    lines.append("Fix each one or tell the cook it cannot be done and why. Do not carry on as "
                 "though the call had worked.")
    return "\n".join(lines)


def to_list(session: Session) -> List[Dict[str, Any]]:
    return [{"tool": f.tool, "reason": f.reason, "at": f.at.isoformat(), "args": f.args}
            for f in session.open_failures]
