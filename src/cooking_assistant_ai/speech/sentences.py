"""Sentence streaming (Phase 5): turn a token stream into TTS-sized sentences early,
so the first audio plays while the model is still generating.

A sentence is only emitted once whitespace *follows* its terminator, so a chunk that
ends in "1." (with "5 minutes" still to come) is never cut in half. Call flush() at
the end of the turn for the tail.
"""
from __future__ import annotations

import re
from typing import List

_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_ABBREV = re.compile(r"\b(e\.g|i\.e|vs|etc|approx|tbsp|tsp|oz|lb|no|st|mr|mrs|dr)\.$", re.I)


class SentenceSplitter:
    def __init__(self, min_chars: int = 12):
        self.buf = ""
        self.min_chars = min_chars

    def feed(self, text: str) -> List[str]:
        self.buf += text
        out: List[str] = []
        pos = 0
        while True:
            m = _BOUNDARY.search(self.buf, pos)
            if not m:
                break
            head = self.buf[:m.start()].strip()
            # Keep abbreviations and very short fragments attached to what follows.
            if not head or _ABBREV.search(head) or len(head) < self.min_chars:
                pos = m.end()
                continue
            out.append(head)
            self.buf = self.buf[m.end():]
            pos = 0
        return out

    def flush(self) -> List[str]:
        tail = self.buf.strip()
        self.buf = ""
        return [tail] if tail else []
