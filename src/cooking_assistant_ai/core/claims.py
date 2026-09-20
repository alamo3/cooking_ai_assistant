"""Claim checking (spec Phase 3, code-side).

The model sometimes *says* it did something ("timer's on for 12 minutes") without calling the
tool. State only changes through tools, so a claim is justified only if a matching tool call
succeeded in the same turn (or, for timer status remarks, a timer really is running).

Detection is regex over sentences, but mood is checked before wording: questions and offers
are dropped first, which is what lets the rules match an assertion in any person. The original
rules all required "I've", which is how Gemini phrased everything. DeepSeek says "Swapped -",
"Plan's set", "Rice is marked done" — so on the day the model changed, claim checking quietly
stopped catching two thirds of claims and nothing reported it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Optional, Set, Tuple  # noqa: F401

from cooking_assistant_ai.model.types import Session

# "Shall I set a timer?" and "Timer's set" differ by mood, not by the words in them, so mood
# is what has to be tested first. Without this guard, dropping the first-person requirement
# would flag every offer the assistant makes.
_NOT_A_CLAIM = re.compile(
    r"\?\s*$"
    r"|\b(?:shall i|should i|do you want|would you like|want me to|shall we)\b"
    # "I'll set a timer" is a commitment, not an offer, and saying it without calling the
    # tool is precisely the drift being guarded against, so it stays a claim.
    r"|\b(?:i can|i could|let me know|if you want|if you like|when you're ready)\b",
    re.I)

RULES: List[Tuple[str, "re.Pattern[str]", FrozenSet[str]]] = [
    ("timer",
     re.compile(r"\b(?:set|start|started|starting|put on)\b[^.]{0,40}\b(?:timer|alarm|countdown)\b"
                r"|\b(?:timer|alarm|countdown)(?:'s| is| has been|s are| are)?\s*"
                r"(?:set|on|going|started|running|ticking)\b"
                r"|\bon the (?:timer|clock)\b", re.I),
     frozenset({"set_timer"})),
    ("complete",
     re.compile(r"\b(?:marked|ticked off|checked off|logged|recorded|noted down|crossed off)\b"
                r"|\b(?:is|are)\s+(?:now\s+)?(?:marked\s+)?(?:done|complete|completed)\b"
                r"|\bmark(?:ing)?\s+(?:that|it|the\s+\w+)\s+(?:as\s+)?done\b", re.I),
     frozenset({"mark_complete", "skip_step", "start_task", "complete_prep"})),
    ("started",
     re.compile(r"\b(?:started|kicked off|begun|began)\b"
                r"|\b(?:is|are|'s)\s+(?:now\s+)?(?:under way|underway)\b"
                r"|\b(?:is|are|'s)\s+(?:now\s+)?"
                r"(?:roasting|simmering|boiling|frying|searing|baking)\b", re.I),
     frozenset({"start_task", "mark_complete"})),
    ("remember",
     re.compile(r"\b(?:noted|remember|remembered|remembering|jotted|made a note|noting that)\b"
                r"|\bkeep(?:ing)? (?:that|this) in mind\b", re.I),
     # "Noted" asserts only that something was written down, so any successful write settles
     # it. The failure this catches is saying "noted" and calling nothing at all.
     frozenset({"remember", "add_note", "set_diet", "add_stock", "mark_complete",
                "complete_prep", "skip_step", "start_task", "substitute"})),
    ("plan",
     re.compile(r"\b(?:added|scheduled|rescheduled|moved|replanned|removed)\b[^.]{0,40}"
                r"\b(?:plan|timeline|schedule|task)\b"
                r"|\b(?:plan|timeline|schedule)(?:'s| is|s are| are)?\s*"
                r"(?:set|built|ready|done|complete|in place|sorted)\b"
                r"|\beverything(?:'s| is)\s+(?:planned|scheduled|set up|sorted)\b", re.I),
     frozenset({"add_task", "move_task", "remove_task", "replan", "set_target_plating"})),
    ("recipe",
     re.compile(r"\b(?:swapped|substituted|subbed|scaled|doubled|halved|skipped)\b"
                r"|\b(?:recipe|it)(?:'s| is| now)\s+(?:uses|using|got)\b"
                r"|\bthat(?:'s| is) the recipe now\b", re.I),
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
    if _NOT_A_CLAIM.search(sentence):
        return out
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


# --------------------------------------------------------------------------- omissions
#
# The mirror of a claim: the cook says something that implies the world changed and the
# model does not record it, so the plan silently drifts from the kitchen. Detection is
# deliberately conservative, and the result is a nudge to the model rather than an
# automatic tool call: deciding what the cook meant is the model's job, not a regex's.

_QUESTION = re.compile(r"^\s*(what|when|where|which|who|why|how|is|are|do|does|did|can|could|"
                       r"should|shall|will|would|am|was|were|have|has|any|remind|tell)\b|\?\s*$", re.I)

_OMISSION_RULES: List[Tuple[str, "re.Pattern[str]", FrozenSet[str]]] = [
    ("started",
     re.compile(r"\b(?:it'?s|they'?re|that'?s|is|are)\s+(?:in|on)\s+(?:the\s+)?(?:oven|stove|stovetop|heat|"
                r"burner|pan|air fryer|grill)\b"
                r"|\b(?:going|goes|putting|put|puts)\s+(?:it|them|that|the \w+)?\s*(?:in|on)(?:to)?\s+(?:the\s+)?"
                r"(?:oven|stove|stovetop|heat|burner|pan|air fryer|grill)\b"
                r"|\b(?:the\s+)?\w+\s+(?:is|are)\s+(?:now\s+)?(?:on|in|simmering|boiling|roasting|frying|"
                r"searing|baking|cooking)\b", re.I),
     frozenset({"advance_step", "start_task", "mark_complete", "set_timer"})),
    ("done",
     # The verb list is open-ended by nature - a recorded cook said "I've mashed the tofu",
     # which none of these covered - so it also accepts any past-tense verb after "I've".
     re.compile(r"\b(?:i'?ve|i have|just)\s+(?:done|finished|completed|chopped|diced|sliced|minced|seared|"
                r"rinsed|washed|peeled|trimmed|seasoned|prepped|prepared|mixed|added|flipped|drained|"
                r"plated|served|started|put|mashed|crumbled|grated|stirred|poured|tipped|dumped)\b"
                r"|\b(?:i'?ve|i have)\s+\w+ed\b"
                r"|\b(?:that'?s|it'?s|they'?re|we'?re|those are|these are)\s+(?:done|finished|ready|complete)\b"
                r"|\b(?:finished|done with)\s+(?:the\s+)?\w+"
                # "okay, I'm ready for the spices" says the step before them is behind us.
                r"|\bi'?m\s+ready\s+for\b", re.I),
     frozenset({"advance_step", "mark_complete", "complete_prep", "start_task", "skip_step"})),
    ("skipped",
     re.compile(r"\b(?:i'?m\s+)?(?:skipping|skip|leaving out|not doing|no)\s+(?:the\s+)?step\b"
                r"|\bskip(?:ping)?\s+(?:the\s+)?\w+\s+step\b", re.I),
     frozenset({"skip_step", "mark_complete"})),
]


@dataclass(frozen=True)
class Omission:
    kind: str
    utterance: str
    needs: FrozenSet[str]
    # Set when the giveaway was the model's own reply rather than something the cook said,
    # where quoting "the cook said ..." back at it would be nonsense.
    detail: Optional[str] = None

    def describe(self) -> str:
        expected = " or ".join(sorted(self.needs))
        if self.detail:
            return f"{self.detail} (expected {expected})"
        return f'the cook said "{self.utterance.strip()}" (expected {expected})'


def implied_state_change(utterance: str) -> Optional[Omission]:
    """A statement by the cook that the world moved on. Questions never count."""
    text = utterance.strip()
    if not text or text.startswith("[SYSTEM]") or _QUESTION.match(text):
        return None
    for kind, pattern, needs in _OMISSION_RULES:
        if pattern.search(text):
            return Omission(kind, text, needs)
    return None


def missed_state_change(utterance: str, tools_ok: Iterable[str]) -> Optional[Omission]:
    omission = implied_state_change(utterance)
    if omission is None or omission.needs & set(tools_ok):
        return None
    return omission


def omission_prompt(omission: Omission) -> str:
    if omission.detail:
        return (
            "[SYSTEM] " + omission.describe() + ". The tablet is still showing them the earlier "
            "one. If you are sending them to a new step, call advance_step for it. If they "
            "finished something out of order, call mark_complete. If you were only "
            "describing what is coming, or are not sure they have done it, ask them. Then "
            "reply to the cook as normal."
        )
    return (
        "[SYSTEM] That turn changed nothing in the state, but " + omission.describe() + ". "
        "If the kitchen really moved on, record it now with the right tool so the plan stays "
        "accurate. If it did not, or you are unsure what they meant, ask them a short question. "
        "Then reply to the cook as normal."
    )


def correction_prompt(claims: List[Claim]) -> str:
    listed = "; ".join(c.describe() for c in claims)
    return (
        "[SYSTEM] Your reply claimed an action that did not happen: " + listed + ". "
        "Nothing changes unless a tool call succeeds. Either make the tool call now, or restate "
        "what you said without claiming it was done. Reply with only the corrected sentence(s)."
    )
