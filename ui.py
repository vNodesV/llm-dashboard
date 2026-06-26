"""
ui.py — curses TUI renderer for llm-watch

Layout (top-down, dynamic height):
  ┌─ llama.cpp Monitor ─────────────── HH:MM:SS ─┐
  │  STATUS / MODEL / REQUEST / UPTIME            │
  │  Decode + bar + averages + peak               │
  │  Prefill / Prompt / Output / Context          │
  ├─ System ─────────────────────────────────────┤
  │  CPU / Load / RAM / RSS                       │
  ├─ Cores ──────────────────────────────────────┤
  │  per-core bars (wraps to fit width)           │
  ├─ History ─────────────────────────────────────┤
  │  tok/s sparkline                              │
  ├─ Last Request ─────────────────────────────────┤
  │  prefill / decode / tokens / duration         │
  ├───────────────────────────────────────────────┤
  │  q Quit   r Reset   c Clear   l Log View      │
  └───────────────────────────────────────────────┘
"""

import curses
import signal
import time
import math
from typing import Optional

from config import Config
from history import bar
from metrics import MetricsSnapshot


# ── Color pair IDs ─────────────────────────────────────────────────────────────
CP_DEFAULT  = 0
CP_TITLE    = 1
CP_GOOD     = 2   # green — above thresh_good
CP_WARN     = 3   # yellow — between thresh_warn and thresh_good
CP_BAD      = 4   # red — below thresh_warn
CP_STATUS   = 5   # cyan — status line
CP_DIM      = 6   # dim white — labels
CP_SECTION  = 7   # bold white — section headers
CP_BAR_FILL = 8   # blue — bar fill


def _fmt_uptime(s: Optional[float]) -> str:
    if s is None:
        return "—"
    s = int(s)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    parts = []
    if d:
        parts.append(f"{d}d")
    if h or d:
        parts.append(f"{h}h")
    parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def _fmt_tps(v: Optional[float], prec: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:.{prec}f}"


def _fmt_gb(v: float) -> str:
    return f"{v:.1f}"


class Dashboard:
    """
    Manages the curses screen. Call .run() — blocks until quit.
    Handles SIGWINCH for resize.
    """

    def __init__(self, collector, cfg: Config):
        self._collector = collector
        self._cfg = cfg
        self._log_view = False
        self._resize_pending = False

    def run(self, stdscr) -> None:
        self._stdscr = stdscr
        self._setup_colors()
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(int(self._cfg.refresh_interval * 1000))

        # SIGWINCH — set flag; redraw on next tick
        signal.signal(signal.SIGWINCH, self._on_resize)

        while True:
            if self._resize_pending:
                curses.endwin()
                stdscr.refresh()
                self._resize_pending = False

            try:
                key = stdscr.getch()
            except curses.error:
                key = -1

            if key in (ord("q"), ord("Q"), 27):  # q or ESC
                break
            elif key in (ord("r"), ord("R")):
                self._collector.reset_stats()
            elif key in (ord("c"), ord("C")):
                self._collector.clear_history()
            elif key in (ord("l"), ord("L")):
                self._log_view = not self._log_view

            snap = self._collector.snapshot
            try:
                self._draw(snap)
            except curses.error:
                pass  # terminal too small — skip frame

    def _on_resize(self, signum, frame):
        self._resize_pending = True

    def _setup_colors(self) -> None:
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(CP_TITLE,    curses.COLOR_CYAN,    -1)
        curses.init_pair(CP_GOOD,     curses.COLOR_GREEN,   -1)
        curses.init_pair(CP_WARN,     curses.COLOR_YELLOW,  -1)
        curses.init_pair(CP_BAD,      curses.COLOR_RED,     -1)
        curses.init_pair(CP_STATUS,   curses.COLOR_CYAN,    -1)
        curses.init_pair(CP_DIM,      curses.COLOR_WHITE,   -1)
        curses.init_pair(CP_SECTION,  curses.COLOR_WHITE,   -1)
        curses.init_pair(CP_BAR_FILL, curses.COLOR_BLUE,    -1)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _addstr(self, row: int, col: int, text: str, attr: int = 0) -> None:
        h, w = self._stdscr.getmaxyx()
        if row < 0 or row >= h:
            return
        # Clip text to available width
        avail = w - col
        if avail <= 0:
            return
        text = text[:avail]
        try:
            self._stdscr.addstr(row, col, text, attr)
        except curses.error:
            pass

    def _hline(self, row: int, ch: str = "─") -> None:
        h, w = self._stdscr.getmaxyx()
        if row < 0 or row >= h:
            return
        try:
            self._stdscr.hline(row, 0, ch, w)
        except curses.error:
            pass

    def _section_header(self, row: int, title: str) -> None:
        _, w = self._stdscr.getmaxyx()
        self._hline(row, "─")
        label = f" {title} "
        attr = curses.color_pair(CP_SECTION) | curses.A_BOLD
        self._addstr(row, 2, label, attr)

    def _tps_color(self, tps: Optional[float]) -> int:
        if tps is None:
            return curses.color_pair(CP_DIM)
        if tps >= self._cfg.thresh_good:
            return curses.color_pair(CP_GOOD) | curses.A_BOLD
        if tps >= self._cfg.thresh_warn:
            return curses.color_pair(CP_WARN) | curses.A_BOLD
        return curses.color_pair(CP_BAD) | curses.A_BOLD

    def _pct_color(self, pct: float) -> int:
        if pct >= 90:
            return curses.color_pair(CP_BAD)
        if pct >= 70:
            return curses.color_pair(CP_WARN)
        return curses.color_pair(CP_GOOD)

    # ── main draw ─────────────────────────────────────────────────────────────

    def _draw(self, snap: MetricsSnapshot) -> None:
        stdscr = self._stdscr
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        if self._log_view:
            self._draw_log(snap, h, w)
        else:
            self._draw_main(snap, h, w)

        stdscr.noutrefresh()
        curses.doupdate()

    # ── log view ─────────────────────────────────────────────────────────────

    def _draw_log(self, snap: MetricsSnapshot, h: int, w: int) -> None:
        self._section_header(0, f"Log View — {snap.source.upper()}")
        lines = snap.log_tail[-(h - 3):]
        for i, line in enumerate(lines):
            self._addstr(i + 1, 1, line[:w - 2], curses.color_pair(CP_DIM))
        self._hline(h - 2)
        self._addstr(h - 1, 1,
                     "l Back   q Quit",
                     curses.color_pair(CP_DIM))

    # ── main view ─────────────────────────────────────────────────────────────

    def _draw_main(self, snap: MetricsSnapshot, h: int, w: int) -> None:
        row = 0

        # ── Title bar ─────────────────────────────────────────────────────────
        ts_str = time.strftime("%H:%M:%S")
        title = " llama.cpp Monitor"
        self._hline(row, "─")
        self._addstr(row, 2, title, curses.color_pair(CP_TITLE) | curses.A_BOLD)
        self._addstr(row, w - len(ts_str) - 2, ts_str,
                     curses.color_pair(CP_DIM))
        row += 1

        # ── Status row ────────────────────────────────────────────────────────
        status_sym = {
            "GENERATING": ("● GENERATING", curses.color_pair(CP_GOOD) | curses.A_BOLD),
            "IDLE":       ("○ IDLE",       curses.color_pair(CP_DIM)),
            "OFFLINE":    ("✗ OFFLINE",    curses.color_pair(CP_BAD)  | curses.A_BOLD),
            "ERROR":      ("⚠ ERROR",      curses.color_pair(CP_BAD)  | curses.A_BOLD),
        }.get(snap.status, (snap.status, curses.color_pair(CP_DIM)))

        model_name = snap.model_name or "—"
        if len(model_name) > 28:
            model_name = model_name[:25] + "…"

        self._addstr(row, 2,  "STATUS  ", curses.color_pair(CP_DIM))
        self._addstr(row, 10, status_sym[0], status_sym[1])
        self._addstr(row, w // 2, "MODEL  ", curses.color_pair(CP_DIM))
        self._addstr(row, w // 2 + 7, model_name,
                     curses.color_pair(CP_SECTION) | curses.A_BOLD)
        row += 1

        req_str = f"#{snap.request_count}" if snap.request_count else "—"
        uptime_str = _fmt_uptime(snap.uptime_wall_s)
        self._addstr(row, 2,  "REQUEST ", curses.color_pair(CP_DIM))
        self._addstr(row, 10, req_str,    curses.color_pair(CP_DIM))
        self._addstr(row, w // 2, "UPTIME ", curses.color_pair(CP_DIM))
        self._addstr(row, w // 2 + 7, uptime_str, curses.color_pair(CP_DIM))
        row += 1

        # ── Decode metrics ────────────────────────────────────────────────────
        row += 1  # blank
        bar_w = min(self._cfg.sparkline_width, w - 30)
        tps = snap.decode_tps
        tps_str = f"{_fmt_tps(tps)} tok/s"
        tps_attr = self._tps_color(tps)

        # Decode row with filled bar
        bar_frac = 0.0
        if tps is not None and snap.peak_tps:
            bar_frac = min(1.0, tps / snap.peak_tps)
        bar_str = bar(bar_frac, width=bar_w, full="█", empty="░")

        self._addstr(row, 2,  "Decode  ", curses.color_pair(CP_DIM))
        self._addstr(row, 10, f"{tps_str:<16}", tps_attr)
        self._addstr(row, 27, bar_str, curses.color_pair(CP_BAR_FILL) | curses.A_BOLD)
        row += 1

        avg_short = snap.avg_short
        avg_mid   = snap.avg_mid
        avg_long  = snap.avg_long
        peak      = snap.peak_tps

        self._addstr(row, 2, "3s Avg  ", curses.color_pair(CP_DIM))
        self._addstr(row, 10, _fmt_tps(avg_short), self._tps_color(avg_short))
        row += 1
        self._addstr(row, 2, "1m Avg  ", curses.color_pair(CP_DIM))
        self._addstr(row, 10, _fmt_tps(avg_mid), self._tps_color(avg_mid))
        row += 1
        if avg_long is not None:
            self._addstr(row, 2, "5m Avg  ", curses.color_pair(CP_DIM))
            self._addstr(row, 10, _fmt_tps(avg_long), self._tps_color(avg_long))
            row += 1
        self._addstr(row, 2, "Peak    ", curses.color_pair(CP_DIM))
        self._addstr(row, 10, _fmt_tps(peak), self._tps_color(peak))
        row += 1

        row += 1  # blank

        # Prefill / tokens / context
        self._addstr(row, 2, "Prefill ", curses.color_pair(CP_DIM))
        pfill = snap.prefill_tps
        self._addstr(row, 10,
                     f"{_fmt_tps(pfill, 0)} tok/s" if pfill else "—",
                     curses.color_pair(CP_GOOD) if pfill else curses.color_pair(CP_DIM))
        row += 1

        self._addstr(row, 2,  "Prompt  ", curses.color_pair(CP_DIM))
        self._addstr(row, 10, f"{snap.n_prompt:,} tokens" if snap.n_prompt else "—",
                     curses.color_pair(CP_DIM))
        row += 1

        self._addstr(row, 2,  "Output  ", curses.color_pair(CP_DIM))
        self._addstr(row, 10, f"{snap.n_output:,} tokens" if snap.n_output else "—",
                     curses.color_pair(CP_DIM))
        row += 1

        ctx_used  = snap.n_ctx_used
        ctx_total = snap.n_ctx_total
        if ctx_total:
            ctx_str = f"{ctx_used:,} / {ctx_total:,}"
            ctx_frac = ctx_used / ctx_total
            ctx_attr = (curses.color_pair(CP_BAD) if ctx_frac > 0.9
                        else curses.color_pair(CP_WARN) if ctx_frac > 0.75
                        else curses.color_pair(CP_DIM))
        else:
            ctx_str = "—"
            ctx_attr = curses.color_pair(CP_DIM)

        self._addstr(row, 2,  "Context ", curses.color_pair(CP_DIM))
        self._addstr(row, 10, ctx_str, ctx_attr)
        row += 1

        # ── System section ────────────────────────────────────────────────────
        row += 1
        self._section_header(row, "System")
        row += 1

        cpu = snap.cpu_pct
        self._addstr(row, 2, "CPU    ", curses.color_pair(CP_DIM))
        self._addstr(row, 9, f"{cpu:.1f}%", self._pct_color(cpu))
        row += 1

        self._addstr(row, 2, "Load   ", curses.color_pair(CP_DIM))
        self._addstr(row, 9, f"{snap.load_avg:.1f}", curses.color_pair(CP_DIM))
        row += 1

        self._addstr(row, 2, "RAM    ", curses.color_pair(CP_DIM))
        ram_str = f"{_fmt_gb(snap.ram_used_gb)} / {_fmt_gb(snap.ram_total_gb)} GB"
        ram_pct = (snap.ram_used_gb / snap.ram_total_gb * 100) if snap.ram_total_gb else 0
        self._addstr(row, 9, ram_str, self._pct_color(ram_pct))
        row += 1

        if snap.model_rss_gb > 0:
            self._addstr(row, 2, "RSS    ", curses.color_pair(CP_DIM))
            self._addstr(row, 9, f"{_fmt_gb(snap.model_rss_gb)} GB",
                         curses.color_pair(CP_DIM))
            row += 1

        # ── Per-core CPU ──────────────────────────────────────────────────────
        cores = snap.cpu_per_core
        if cores and row + 4 < h:
            row += 0
            self._section_header(row, "Cores")
            row += 1

            core_bar_w = max(10, min(40, w - 12))
            cols_per_row = max(1, w // (core_bar_w + 8))

            for i, pct in enumerate(cores):
                col_idx = i % cols_per_row
                if col_idx == 0 and i > 0:
                    row += 1
                if row >= h - 4:
                    break
                col_off = col_idx * (core_bar_w + 8)
                b = bar(pct / 100, width=core_bar_w, full="█", empty="░")
                pct_label = f"{pct:4.0f}%"
                self._addstr(row, col_off + 2, pct_label, self._pct_color(pct))
                self._addstr(row, col_off + 8, b, curses.color_pair(CP_BAR_FILL))

            row += 2

        # ── Decode history sparkline ───────────────────────────────────────────
        if snap.decode_sparkline and row + 3 < h:
            self._section_header(row, "Decode History")
            row += 1
            spark = snap.decode_sparkline[: w - 4]
            self._addstr(row, 2, spark, curses.color_pair(CP_GOOD) | curses.A_BOLD)
            row += 2

        # ── Last request ──────────────────────────────────────────────────────
        last = snap.last_request
        if last and row + 6 < h:
            self._section_header(row, "Last Request")
            row += 1

            self._addstr(row, 2, "Prefill  ", curses.color_pair(CP_DIM))
            self._addstr(row, 11,
                         f"{_fmt_tps(last.prefill_tps, 0)} tok/s" if last.prefill_tps else "—",
                         curses.color_pair(CP_DIM))
            row += 1

            self._addstr(row, 2, "Decode   ", curses.color_pair(CP_DIM))
            self._addstr(row, 11,
                         f"{_fmt_tps(last.decode_tps)} tok/s" if last.decode_tps else "—",
                         self._tps_color(last.decode_tps))
            row += 1

            self._addstr(row, 2, "Tokens   ", curses.color_pair(CP_DIM))
            self._addstr(row, 11, f"{last.n_output:,}" if last.n_output else "—",
                         curses.color_pair(CP_DIM))
            row += 1

            dur = last.duration_s
            self._addstr(row, 2, "Duration ", curses.color_pair(CP_DIM))
            self._addstr(row, 11,
                         f"{dur:.1f} s" if dur else "—",
                         curses.color_pair(CP_DIM))
            row += 2

        # ── Status/source indicator ────────────────────────────────────────────
        source_label = {
            "prometheus": f"⬡ prometheus  http://{self._cfg.host}:{self._cfg.port}",
            "log":        f"▤ log file",
            "offline":    "✗ no source — searching…",
        }.get(snap.source, snap.source)

        if row < h - 2:
            self._section_header(row, "")
            self._addstr(row, w - len(source_label) - 2, source_label,
                         curses.color_pair(CP_DIM))
            row += 1

        # ── Key bindings bar ──────────────────────────────────────────────────
        if row < h:
            keybinds = "  q Quit    r Reset Stats    c Clear History    l Log View"
            self._addstr(min(row, h - 1), 0, keybinds[:w], curses.color_pair(CP_DIM))
