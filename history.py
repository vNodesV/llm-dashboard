"""
history.py — rolling stats, sparklines, percentile tracking
"""

import time
from collections import deque
from typing import Optional, List, Tuple


# ── Sparkline characters (8 levels) ──────────────────────────────────────────
SPARKS = " ▁▂▃▄▅▆▇█"


def sparkline(values: List[float], width: int = 40) -> str:
    """
    Build a sparkline string from a list of float values.
    Width is the number of characters to render (truncates/pads left).
    Returns empty string if no values.
    """
    if not values:
        return ""
    v = list(values)[-width:]  # take last `width` samples
    lo, hi = min(v), max(v)
    span = hi - lo
    result = []
    for x in v:
        if span == 0:
            idx = 4  # middle
        else:
            idx = int((x - lo) / span * (len(SPARKS) - 1))
        result.append(SPARKS[idx])
    return "".join(result)


def bar(fraction: float, width: int = 20, full: str = "█", empty: str = " ") -> str:
    """
    Render a filled bar of given width.
    fraction in [0.0, 1.0].
    Supports partial last block via ▉▊▋▌▍▎▏.
    """
    fraction = max(0.0, min(1.0, fraction))
    filled = fraction * width
    full_blocks = int(filled)
    partial = filled - full_blocks
    # partial block chars (index 0 = none, 7 = nearly full)
    PARTIALS = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]
    part_char = PARTIALS[int(partial * 8)] if partial > 0.01 else ""
    empty_count = width - full_blocks - (1 if part_char else 0)
    return full * full_blocks + part_char + empty * empty_count


# ── Timed sample ──────────────────────────────────────────────────────────────

class Sample:
    __slots__ = ("ts", "value")

    def __init__(self, value: float):
        self.ts = time.monotonic()
        self.value = value


# ── Rolling history ───────────────────────────────────────────────────────────

class RollingHistory:
    """
    Stores a bounded ring of timed float samples and derives:
    - rolling averages over configurable windows
    - P50/P95/P99 percentiles over all stored samples
    - sparkline for the last N samples
    - peak value
    """

    def __init__(self, maxlen: int = 120):
        self._samples: deque[Sample] = deque(maxlen=maxlen)
        self._peak: Optional[float] = None

    def push(self, value: float) -> None:
        if value is None:
            return
        self._samples.append(Sample(value))
        if self._peak is None or value > self._peak:
            self._peak = value

    def rolling_avg(self, window_s: float) -> Optional[float]:
        """Average of samples within the last window_s seconds."""
        if not self._samples:
            return None
        cutoff = time.monotonic() - window_s
        vals = [s.value for s in self._samples if s.ts >= cutoff]
        return sum(vals) / len(vals) if vals else None

    def latest(self) -> Optional[float]:
        return self._samples[-1].value if self._samples else None

    @property
    def peak(self) -> Optional[float]:
        return self._peak

    def percentile(self, pct: float) -> Optional[float]:
        """Compute percentile (0-100) over all stored samples."""
        if not self._samples:
            return None
        vals = sorted(s.value for s in self._samples)
        idx = (pct / 100) * (len(vals) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(vals) - 1)
        frac = idx - lo
        return vals[lo] * (1 - frac) + vals[hi] * frac

    def values(self) -> List[float]:
        return [s.value for s in self._samples]

    def sparkline(self, width: int = 40) -> str:
        return sparkline(self.values(), width)

    def reset(self) -> None:
        self._samples.clear()
        self._peak = None


# ── Request history ───────────────────────────────────────────────────────────

class RequestRecord:
    """One completed llama-server generation request."""

    __slots__ = (
        "task_id", "prefill_tps", "decode_tps",
        "n_prompt", "n_output", "n_ctx",
        "duration_s", "ts",
    )

    def __init__(
        self,
        task_id: Optional[int],
        prefill_tps: Optional[float],
        decode_tps: Optional[float],
        n_prompt: int,
        n_output: int,
        n_ctx: int,
        duration_s: Optional[float],
    ):
        self.task_id = task_id
        self.prefill_tps = prefill_tps
        self.decode_tps = decode_tps
        self.n_prompt = n_prompt
        self.n_output = n_output
        self.n_ctx = n_ctx
        self.duration_s = duration_s
        self.ts = time.time()


class RequestHistory:
    """Ring buffer of completed requests."""

    def __init__(self, maxlen: int = 50):
        self._records: deque[RequestRecord] = deque(maxlen=maxlen)

    def push(self, record: RequestRecord) -> None:
        self._records.append(record)

    def latest(self) -> Optional[RequestRecord]:
        return self._records[-1] if self._records else None

    def count(self) -> int:
        return len(self._records)

    def total(self) -> int:
        """All-time request count (capped at maxlen if history was lost)."""
        return self.count()

    def reset(self) -> None:
        self._records.clear()
