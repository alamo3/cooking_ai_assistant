"""Claim checking (spec Phase 3, code-side).

The model sometimes *says* it did something ("I've set a timer for 25 minutes") without
calling the tool. Every such sentence is a first-person claim about state, and state only
changes through tools, so a claim is justified only if a matching tool call succeeded in
the same turn (or, for timer status remarks, a timer is actually running).

Detection is regex over sentences; deliberately conservative so that ordinary speech
("there are 15 minutes left on the rice timer") is never flagged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Optional, Set, Tuple

from cooking_assistant_ai.model.types import Session

_I = r"\b(?:i(?:'ve|'ll| have| will| just|'m going to| am going to)?|let me|i'll go ahead and|i've gone ahead and)"

RULES: List[Tuple[str, "re.Pattern[str]", FrozenSet[str]]] = [
    ("timer",
     re.compile(_I + r" (?:set|start|put on|create|get|add)\w* (?:up )?(?:a |the |your |another |that )?(?:\w+[- ]){0,4}(?:timer|alarm|countdown)\b"
                r"|\b(?:timer|alarm|countdown)(?:'s| is| has been|s are| are)? (?:set|on|going|started|running|ticking)\b"
                r"|\bset (?:a|the|your) (?:\w+[- ]){0,4}(?:timer|alarm)\b", re.I),
     frozenset({"set_timer"})),
    ("complete",
     re.compile(_I + r" (?:mark|marked|record|recorded|check|checked off|log|logged)\b", re.I),
     frozenset({"mark_complete", "skip_step", "start_task"})),
    ("started",
     re.compile(_I + r" (?:started|kicked off|begun|began)\b", re.I),
     frozenset({"start_task"})),
    ("remember",
     re.compile(_I + r" (?:remember|remembered|note|noted|make a note|made a note|jot|keep (?:that |this )?in mind)\b", re.I),
     frozenset({"remember", "add_note"})),
    ("plan",
     re.compile(_I + r" (?:add|added|schedule|scheduled|move|moved|reschedule|rescheduled|remove|removed|replan|replanned|"
                r"update|updated|set) (?:\w+[- ]){0,3}(?:plan|timeline|schedule|plating|task|tasks)\b"
                r"|" + _I + r" (?:added|scheduled|rescheduled|moved) (?:the |a |your )?(?:\w+[- ]){0,3}(?:to|on|in) the (?:plan|timeline|schedule)\b"
                r"|\b(?:the |your )?(?:plan|timeline|schedule) (?:is|looks|'s) (?:ready|set|done|complete|built|in place)\b"
                r"|\beverything (?:is|'s) (?:planned|scheduled|set up|on the timeline)\b", re.I),
     frozenset({"add_task", "move_task", "remove_task", "replan", "set_target_plating"})),
    ("recipe",
     re.compile(_I + r" (?:swap|swapped|substitute|substituted|scale|scaled|double|doubled|halve|halved|skip|skipped|"
                r"update|updated) (?:\w+[- ]){0,3}(?:recipe|ingredient|ingredients|substitution|step|amounts)\b"
                r"|" + _I + r" (?:swapped|substituted|doubled|halved|scaled)\b", re.I),
     frozenset({"substitute", "scale", "add_note", "add_step", "skip_step"})),
]

_STATUS_REMARK = re.compile(r"\b(?:still|left|remaining|running|going|has|about|minutes?|seconds?|until)\b", re.I)
_STOPWORDS = {"the", "a", "an", "your", "for", "on", "is", "set", "timer", "and", "to", "of", "it", "that", "this"}


@dataclass(frozen=True)
class Claim:
    kind: str
    sentence: str
    needs: FrozenSet[str]

    def describe(self) -> str:
        return f'"{self.sentence.strip()}" (needs a successful {" or ".join(sorted(self.needs))} call)'


def find_claims(sentence: str) -> List[Claim]:
    out: List[Claim] = []
    for kind, pattern, needs in RULES:
        if pattern.search(sentence):
            out.append(Claim(kind, sentence, needs))
    return out


def _words(text: str) -> Set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w not in _STOPWORDS and len(w) > 2}


def is_justified(claim: Claim, tools_ok: Iterable[str], session: Optional[Session] = None) -> bool:
    if claim.needs & set(tools_ok):
        return True
    if claim.kind == "plan" and session is not None:
        # "The plan is ready" is a fair status remark once tasks exist, and a lie when none do.
        return bool(session.tasks)
    if claim.kind == "timer" and session is not None:
        # "Your rice timer is still running" is a status remark, fine if such a timer exists.
        running = session.running_timers()
        if running and _STATUS_REMARK.search(claim.sentence):
            words = _words(claim.sentence)
            return any(_words(t.label) & words for t in running) or "timer is" in claim.sentence.lower()
    return False


def unjustified_claims(sentence: str, tools_ok: Iterable[str], session: Optional[Session] = None) -> List[Claim]:
    tools = set(tools_ok)
    return [c for c in find_claims(sentence) if not is_justified(c, tools, session)]


def correction_prompt(claims: List[Claim]) -> str:
    listed = "; ".join(c.describe() for c in claims)
    return (
        "[SYSTEM] Your reply claimed an action that did not happen: " + listed + ". "
        "Nothing changes unless a tool call succeeds. Either make the tool call now, or restate "
        "what you said without claiming it was done. Reply with only the corrected sentence(s)."
    )
