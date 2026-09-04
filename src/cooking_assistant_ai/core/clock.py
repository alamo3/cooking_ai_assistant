"""Injectable wall clock so tests (and the REPL) can move time without sleeping."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional


class Clock:
    def __init__(self, fixed: Optional[datetime] = None):
        self._fixed = fixed
        self._offset = timedelta(0)

    def now(self) -> datetime:
        if self._fixed is not None:
            return self._fixed + self._offset
        return datetime.now() + self._offset

    def advance(self, seconds: float) -> None:
        self._offset += timedelta(seconds=seconds)

    def set(self, when: datetime) -> None:
        self._fixed = when
        self._offset = timedelta(0)

    @property
    def is_simulated(self) -> bool:
        return self._fixed is not None
