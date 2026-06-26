"""
config.py — llm-watch configuration
All tunable settings in one place.
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class Config:
    # ── llama-server connection ───────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 8080

    # ── log file (optional; auto-detected if blank) ───────────────────────────
    log_file: Optional[str] = None

    # ── data collection ───────────────────────────────────────────────────────
    refresh_hz: float = 2.0          # UI refresh rate (Hz)
    metrics_poll_hz: float = 4.0     # metrics poll rate (Hz) — faster than UI
    prometheus_path: str = "/metrics"
    prefer_prometheus: bool = True   # prefer /metrics over log parsing when available

    # ── history ───────────────────────────────────────────────────────────────
    history_len: int = 120           # number of samples in sparkline history
    avg_short_s: float = 15.0       # short rolling average window (seconds) — wider than slot_print_timing interval (~3s)
    avg_mid_s: float = 60.0         # mid rolling average window (seconds)
    avg_long_s: float = 300.0       # long rolling average window (seconds)

    # ── display ───────────────────────────────────────────────────────────────
    color: bool = True
    sparkline_width: int = 40        # max sparkline bar width for current tok/s

    # ── thresholds (tok/s) for color coding ───────────────────────────────────
    thresh_warn: float = 10.0        # below → yellow
    thresh_good: float = 20.0        # above → green

    # ── CSV recording ─────────────────────────────────────────────────────────
    csv_out: Optional[str] = None    # path to write CSV, None to disable

    # ── log view ──────────────────────────────────────────────────────────────
    log_lines_max: int = 200         # lines kept in log ring buffer

    # ── auto-detect common log paths ──────────────────────────────────────────
    _auto_log_paths: list = field(default_factory=lambda: [
        "/tmp/llama.log",
        "/tmp/llama-server.log",
        "/var/log/llama-server.log",
        os.path.expanduser("~/llama-server.log"),
        os.path.expanduser("~/.local/log/llama-server.log"),
    ])

    def prometheus_url(self) -> str:
        return f"http://{self.host}:{self.port}{self.prometheus_path}"

    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}/health"

    def resolve_log_file(self) -> Optional[str]:
        """Return log_file if set, else first auto-detected path that exists."""
        if self.log_file:
            return self.log_file if os.path.isfile(self.log_file) else None
        for p in self._auto_log_paths:
            if os.path.isfile(p):
                return p
        return None

    @property
    def refresh_interval(self) -> float:
        return 1.0 / self.refresh_hz

    @property
    def metrics_interval(self) -> float:
        return 1.0 / self.metrics_poll_hz
