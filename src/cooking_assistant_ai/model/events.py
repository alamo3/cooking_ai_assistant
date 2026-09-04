"""Events consumed by the orchestrator. Two producers (transport + timers), one queue."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from cooking_assistant_ai.model.types import Timer


@dataclass
class UserUtterance:
    text: str


@dataclass
class TimerFired:
    timer: Timer


@dataclass
class IdleTick:
    since_last_turn_s: int


@dataclass
class SystemPrompt:
    """An instruction from the app (not the cook), answered out loud like a normal turn."""
    text: str


Event = Union[UserUtterance, TimerFired, IdleTick, SystemPrompt]
