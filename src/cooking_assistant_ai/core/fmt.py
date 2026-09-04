"""Formatting + parsing helpers shared by renderers and tools."""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

_FRACTIONS = {
    0.25: "1/4", 0.33: "1/3", 0.5: "1/2", 0.67: "2/3", 0.75: "3/4",
    0.125: "1/8", 0.375: "3/8", 0.625: "5/8", 0.875: "7/8",
}


def fmt_time(dt: Optional[datetime]) -> str:
    if dt is None:
        return "--:--"
    return dt.strftime("%I:%M %p").lstrip("0")


def fmt_window(start: Optional[datetime], end: Optional[datetime]) -> str:
    if start is None or end is None:
        return "--:-- to --:--"
    return f"{start.strftime('%I:%M').lstrip('0')}-{end.strftime('%I:%M').lstrip('0')}"


def fmt_dur(seconds: float) -> str:
    """33m, 1h05m, 45s. Negative durations are shown as overdue with a leading '-'."""
    neg = seconds < 0
    s = int(round(abs(seconds)))
    if s < 60:
        out = f"{s}s"
    elif s < 3600:
        m, rem = divmod(s, 60)
        out = f"{m}m" if rem < 30 or m >= 10 else f"{m}m{rem:02d}s"
    else:
        h, rem = divmod(s, 3600)
        m = rem // 60
        out = f"{h}h" if m == 0 else f"{h}h{m:02d}m"
    return ("-" + out) if neg else out


def fmt_amount(amount: float) -> str:
    """2, 1.5, 1/4, 1 1/2 -- friendly for both eyes and TTS."""
    if amount == int(amount):
        return str(int(amount))
    whole = int(amount)
    frac = round(amount - whole, 3)
    for key, text in _FRACTIONS.items():
        if abs(frac - key) < 0.02:
            return f"{whole} {text}" if whole else text
    return f"{amount:.2f}".rstrip("0").rstrip(".")


def fmt_ingredient(name: str, amount: float, unit: Optional[str]) -> str:
    qty = fmt_amount(amount)
    return f"{qty} {unit} {name}" if unit else f"{qty} {name}"


_TIME_FORMATS = ["%I:%M %p", "%I:%M%p", "%I %p", "%I%p", "%H:%M", "%H"]


def parse_clock_time(text: str, now: datetime) -> Optional[datetime]:
    """Parse '7:15 PM', '7:15pm', '19:15', '7 pm' as today's date. Returns None on failure."""
    raw = text.strip().upper().replace(".", "")
    raw = re.sub(r"\s+", " ", raw)
    for f in _TIME_FORMATS:
        try:
            t = datetime.strptime(raw, f)
        except ValueError:
            continue
        return now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
    return None


def parse_duration(text: str) -> Optional[int]:
    """'10m', '1h30m', '90s', '45 minutes', '1.5 hours' -> seconds."""
    raw = text.strip().lower()
    if raw.isdigit():
        return int(raw) * 60
    total = 0.0
    found = False
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes|s|sec|secs|second|seconds)", raw):
        found = True
        n = float(num)
        if unit.startswith("h"):
            total += n * 3600
        elif unit.startswith("m"):
            total += n * 60
        else:
            total += n
    return int(total) if found else None


def floor_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


def plus(dt: datetime, seconds: float) -> datetime:
    return dt + timedelta(seconds=seconds)
