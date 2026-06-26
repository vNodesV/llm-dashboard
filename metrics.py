"""
metrics.py — llama.cpp metrics collector

Two modes (auto-selected based on availability):
  1. Prometheus /metrics polling  (preferred — structured, low latency)
  2. Log file tail                (fallback — line-by-line parse, near-zero CPU)

Both modes update a shared MetricsSnapshot that the UI reads.
Thread-safe: snapshot is replaced atomically (object swap + lock).
"""

import os
import re
import sys
import time
import threading
import csv
from dataclasses import dataclass, field
from typing import Optional, List
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

import psutil

from config import Config
from history import RollingHistory, RequestHistory, RequestRecord


# ── Snapshot (immutable read view for UI) ─────────────────────────────────────

@dataclass
class MetricsSnapshot:
    # connection
    source: str = "?"                # "prometheus" | "log" | "offline"
    connected: bool = False

    # model (populated from /models on startup)
    model_name: str = ""
    model_size_gb: float = 0.0
    model_params_b: float = 0.0
    n_ctx_train: int = 0
    server_uptime_s: Optional[float] = None

    # generation state
    status: str = "IDLE"             # "IDLE" | "GENERATING" | "ERROR"
    task_id: Optional[int] = None
    request_count: int = 0

    # current decode
    decode_tps: Optional[float] = None
    prefill_tps: Optional[float] = None
    server_tg3s: Optional[float] = None   # tg_3s from slot print_timing (server-computed)

    # rolling averages
    avg_short: Optional[float] = None
    avg_mid: Optional[float] = None
    avg_long: Optional[float] = None
    peak_tps: Optional[float] = None

    # context / tokens
    n_prompt: int = 0
    n_output: int = 0
    n_ctx_used: int = 0
    n_ctx_total: int = 0

    # system
    cpu_pct: float = 0.0
    cpu_per_core: List[float] = field(default_factory=list)
    load_avg: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    model_rss_gb: float = 0.0

    # history (sparkline source)
    decode_history: List[float] = field(default_factory=list)
    decode_sparkline: str = ""

    # last completed request
    last_request: Optional[RequestRecord] = None

    # log view
    log_tail: List[str] = field(default_factory=list)

    # timestamps
    ts: float = 0.0
    uptime_wall_s: float = 0.0      # time since collector started


# ── Log parser ────────────────────────────────────────────────────────────────

class LogParser:
    """
    Incrementally tail a log file and extract llama.cpp metrics.
    Keeps the file handle open; seeks to end-of-file after opening so we
    only process NEW lines — no full re-scan on every poll.
    """

    # llama_print_timings patterns (llama.cpp ≥ b3000)
    RE_EVAL = re.compile(
        r"llama_print_timings:\s+eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s+runs"
        r".*?(\d[\d.]*)\s+tokens per second",
        re.I,
    )
    RE_PROMPT = re.compile(
        r"llama_print_timings:\s+prompt eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s+tokens"
        r".*?(\d[\d.]*)\s+tokens per second",
        re.I,
    )
    RE_TOTAL = re.compile(
        r"llama_print_timings:\s+total time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s+tokens",
        re.I,
    )
    RE_SAMPLE = re.compile(
        r"llama_print_timings:\s+sample time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s+runs",
        re.I,
    )

    # Server slot/status patterns
    # New format: "slot release: id  2 | task 92 | ..."
    # Old format: "slot release ... task_id = 92 ..."
    RE_SLOT_START = re.compile(
        r"slot\s+launch.*?(?:task[_\s]id\s*[=:]\s*|task\s+)(\d+)", re.I
    )
    RE_SLOT_DONE  = re.compile(
        r"slot\s+release.*?(?:task[_\s]id\s*[=:]\s*|task\s+)(\d+)", re.I
    )
    RE_MODEL_PATH = re.compile(r"([^\s/]+\.gguf)", re.I)
    RE_CTX        = re.compile(r"n_ctx\s*=\s*(\d+)", re.I)

    # ── NEW: live timing lines emitted every ~3s during generation ────────────
    # Format: "slot print_timing: id  2 | task 92 | n_decoded =  188, tg =  10.18 t/s, tg_3s =   8.18 t/s"
    # Extract task id and n_decoded first, then pull ALL key=value t/s pairs via findall.
    RE_SLOT_TIMING_HDR = re.compile(
        r"slot\s+print_timing:.*?task\s+(\d+).*?n_decoded\s*=\s*(\d+)",
        re.I,
    )
    # Matches every "key = value t/s" pair on the line (tg, tg_3s, tg_10s, etc.)
    RE_TG_PAIRS = re.compile(r"\b(tg(?:_\d+s)?)\s*=\s*([\d.]+)\s*t/s", re.I)
    # Prefill line (if present): "slot eval: id X | task Y | ... prompt_tg = N t/s"
    RE_SLOT_EVAL = re.compile(
        r"slot\s+(?:eval|prompt).*?task\s+(\d+).*?(?:prompt_tg|pp)\s*=\s*([\d.]+)\s*t/s",
        re.I,
    )

    # Older server format (b3xxx+) JSON timings in log
    RE_NEW_DECODE = re.compile(
        r'"timings".*?"predicted_per_second":\s*([\d.]+)', re.I
    )
    RE_NEW_PROMPT = re.compile(
        r'"timings".*?"prompt_per_second":\s*([\d.]+)', re.I
    )

    def __init__(self, path: str):
        self.path = path
        self._fh = None
        self._pos = 0
        self._ctx_total: int = 0
        self.model_name: str = ""

        # Live state — updated on every slot print_timing line
        self.live_tps: Optional[float] = None    # tg = X t/s (current overall)
        self.live_tg3s: Optional[float] = None   # tg_3s = X t/s (3s rolling avg)
        self.live_n_decoded: int = 0             # n_decoded so far this request
        self.live_prefill_tps: Optional[float] = None

        # Per-request accumulator (for legacy llama_print_timings format)
        self._pending_prompt: Optional[float] = None
        self._pending_eval_tokens: Optional[int] = None
        self._pending_decode: Optional[float] = None

        # completed request data
        self.last_record: Optional[RequestRecord] = None
        self.current_task: Optional[int] = None
        self.is_generating: bool = False

    def _open(self) -> bool:
        try:
            self._fh = open(self.path, "r", encoding="utf-8", errors="replace")
            self._fh.seek(0, 2)  # seek to end — only new lines
            self._pos = self._fh.tell()
            return True
        except OSError:
            return False

    def poll(self) -> List[str]:
        """
        Read any new lines appended since last poll.
        Returns list of raw log lines (for log view).
        Also updates internal state.
        """
        if self._fh is None:
            if not self._open():
                return []

        # Re-open if file was rotated
        try:
            stat = os.stat(self.path)
            if self._fh.fileno() >= 0:
                cur_stat = os.fstat(self._fh.fileno())
                if stat.st_ino != cur_stat.st_ino or stat.st_size < self._pos:
                    self._fh.close()
                    if not self._open():
                        return []
        except OSError:
            return []

        new_lines = []
        try:
            self._fh.seek(self._pos)
            for line in self._fh:
                new_lines.append(line.rstrip("\n"))
                self._parse_line(line)
            self._pos = self._fh.tell()
        except OSError:
            pass

        return new_lines

    def _parse_line(self, line: str) -> None:
        # ── Model name (one-time, from any .gguf reference in log) ───────────
        if not self.model_name:
            m = self.RE_MODEL_PATH.search(line)
            if m:
                self.model_name = m.group(1).removesuffix(".gguf")

        # ── Context size (one-time) ───────────────────────────────────────────
        if not self._ctx_total:
            m = self.RE_CTX.search(line)
            if m:
                n = int(m.group(1))
                # sanity: ignore tiny values (e.g. n_ctx_per_seq = 512)
                if n >= 512:
                    self._ctx_total = n

        # ── NEW FORMAT: slot print_timing — live tok/s every ~3s ─────────────
        # "slot print_timing: id  2 | task 92 | n_decoded = 188, tg = 10.18 t/s, tg_3s = 8.18 t/s"
        # Two-step: header captures task+n_decoded; findall captures all tg_* pairs.
        m = self.RE_SLOT_TIMING_HDR.search(line)
        if m:
            self.current_task   = int(m.group(1))
            self.live_n_decoded = int(m.group(2))
            tg = dict(self.RE_TG_PAIRS.findall(line))
            self.live_tps   = float(tg["tg"])       if "tg"    in tg else None
            self.live_tg3s  = float(tg["tg_3s"])    if "tg_3s" in tg else None
            self.is_generating = True
            return  # dominant signal — skip legacy checks

        # ── NEW FORMAT: slot prefill timing ───────────────────────────────────
        m = self.RE_SLOT_EVAL.search(line)
        if m:
            self.live_prefill_tps = float(m.group(2))
            return

        # ── Slot start / release ──────────────────────────────────────────────
        m = self.RE_SLOT_START.search(line)
        if m:
            self.current_task  = int(m.group(1))
            self.is_generating = True

        m = self.RE_SLOT_DONE.search(line)
        if m:
            task_id = int(m.group(1))
            # Emit completed request record using last known live values
            if self.live_tps is not None or self._pending_decode is not None:
                decode_tps = self.live_tps or self._pending_decode
                n_output   = self.live_n_decoded or (self._pending_eval_tokens or 0)
                self.last_record = RequestRecord(
                    task_id=task_id,
                    prefill_tps=self.live_prefill_tps or self._pending_prompt,
                    decode_tps=decode_tps,
                    n_prompt=0,   # not available in new format without separate eval line
                    n_output=n_output,
                    n_ctx=self._ctx_total,
                    duration_s=(n_output / decode_tps) if decode_tps and n_output else None,
                )
            # Reset live state for next request
            self.live_tps       = None
            self.live_tg3s      = None
            self.live_n_decoded = 0
            self.live_prefill_tps = None
            self._pending_prompt = None
            self._pending_eval_tokens = None
            self._pending_decode = None
            self.is_generating  = False

        # ── LEGACY FORMAT: llama_print_timings block ──────────────────────────
        m = self.RE_PROMPT.search(line)
        if m:
            self._pending_prompt = float(m.group(2))

        m = self.RE_EVAL.search(line)
        if m:
            self._pending_eval_tokens = int(m.group(1))
            self._pending_decode      = float(m.group(2))

        m = self.RE_TOTAL.search(line)
        if m:
            total_toks = int(m.group(1))
            self.last_record = RequestRecord(
                task_id=self.current_task,
                prefill_tps=self._pending_prompt,
                decode_tps=self._pending_decode,
                n_prompt=total_toks - (self._pending_eval_tokens or 0),
                n_output=self._pending_eval_tokens or 0,
                n_ctx=self._ctx_total,
                duration_s=(
                    (self._pending_eval_tokens / self._pending_decode)
                    if self._pending_decode and self._pending_eval_tokens else None
                ),
            )
            self._pending_prompt      = None
            self._pending_eval_tokens = None
            self._pending_decode      = None
            self.is_generating        = False

        # JSON timings (some b3xxx builds)
        m = self.RE_NEW_DECODE.search(line)
        if m:
            self._pending_decode = float(m.group(1))

        m = self.RE_NEW_PROMPT.search(line)
        if m:
            self._pending_prompt = float(m.group(1))


# ── Prometheus parser ─────────────────────────────────────────────────────────

class PrometheusPoller:
    """
    Poll llama-server /metrics and extract key counters.
    Returns a dict of metric_name → float.
    Non-blocking; raises on error so caller can fallback.
    """

    # Known llama-server Prometheus metric names
    KEYS = {
        "llamacpp:tokens_per_second",
        "llamacpp:prompt_tokens_total",
        "llamacpp:generation_tokens_total",
        "llamacpp:kv_cache_usage_ratio",
        "llamacpp:kv_cache_tokens",
        "llamacpp:requests_processing",
        "llamacpp:requests_deferred",
        "llamacpp:requests_pending",
    }

    def __init__(self, url: str, timeout: float = 2.0):
        self.url = url
        self.timeout = timeout

    def poll(self) -> dict:
        """Fetch and parse /metrics. Returns dict of name → float."""
        try:
            with urlopen(self.url, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except (URLError, HTTPError, OSError, TimeoutError):
            raise

        result = {}
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("#"):
                continue
            parts = line.rsplit(None, 1)
            if len(parts) != 2:
                continue
            name_raw, val_raw = parts
            name = name_raw.split("{")[0]  # strip labels
            try:
                result[name] = float(val_raw)
            except ValueError:
                pass
        return result


# ── System metrics ────────────────────────────────────────────────────────────

class ModelsPoller:
    """
    One-shot fetch of /models to get static model metadata:
    name, n_ctx, n_ctx_train, size, n_params.
    Called once on startup; retried until successful.
    """

    def __init__(self, base_url: str, timeout: float = 3.0):
        self.url = base_url.rstrip("/") + "/models"
        self.timeout = timeout

    def fetch(self) -> dict:
        """
        Returns dict with keys:
          model_name, n_ctx, n_ctx_train, size_gb, params_b
        Raises on failure so caller can retry.
        """
        import json
        with urlopen(self.url, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        # llama-server returns both OAI-compat "data" array and "models" array
        # "data" has the richer "meta" sub-object
        result = {}
        entries = body.get("data") or body.get("models") or []
        if not entries:
            return result

        entry = entries[0]
        raw_id = entry.get("id") or entry.get("name") or ""
        # Strip path and .gguf suffix for display
        name = raw_id.split("/")[-1]
        if name.lower().endswith(".gguf"):
            name = name[:-5]
        result["model_name"] = name

        meta = entry.get("meta") or {}
        result["n_ctx"]       = int(meta.get("n_ctx", 0))
        result["n_ctx_train"] = int(meta.get("n_ctx_train", 0))
        result["size_gb"]     = meta.get("size", 0) / 1e9 if meta.get("size") else 0.0
        result["params_b"]    = meta.get("n_params", 0) / 1e9 if meta.get("n_params") else 0.0
        return result


class SystemPoller:
    """Thin wrapper around psutil for system metrics."""

    def __init__(self):
        # Prime psutil's per-CPU interval baseline
        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(percpu=True, interval=None)

    def poll(self) -> dict:
        mem = psutil.virtual_memory()
        load = psutil.getloadavg()
        return {
            "cpu_pct": psutil.cpu_percent(interval=None),
            "cpu_per_core": psutil.cpu_percent(percpu=True, interval=None),
            "load_avg": load[0],
            "ram_used_gb": mem.used / 1e9,
            "ram_total_gb": mem.total / 1e9,
            "model_rss_gb": self._llama_rss(),
        }

    @staticmethod
    def _llama_rss() -> float:
        """Sum RSS of all llama-server / llama.main processes."""
        rss = 0.0
        keywords = ("llama-server", "llama_server", "llama.main", "llama-cli")
        try:
            for proc in psutil.process_iter(["name", "memory_info"]):
                name = (proc.info.get("name") or "").lower()
                if any(k in name for k in keywords):
                    mi = proc.info.get("memory_info")
                    if mi:
                        rss += mi.rss / 1e9
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return rss


# ── Collector (background thread) ─────────────────────────────────────────────

class Collector:
    """
    Background thread that polls metrics and maintains a current snapshot.
    UI reads self.snapshot (lock-protected swap).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._snapshot = MetricsSnapshot(ts=time.time())
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started_at = time.monotonic()

        self._decode_hist = RollingHistory(maxlen=cfg.history_len)
        self._req_hist = RequestHistory()
        self._sys_poller = SystemPoller()
        self._log_parser: Optional[LogParser] = None
        self._prom_poller: Optional[PrometheusPoller] = None
        self._models_poller = ModelsPoller(
            f"http://{cfg.host}:{cfg.port}"
        )
        self._model_meta: dict = {}   # populated on first successful /models fetch
        self._log_tail: List[str] = []
        self._csv_writer = None
        self._csv_file = None
        self._request_counter = 0

        # Counter state for Prometheus rate computation (Δtokens / Δtime)
        self._prev_prom_ts: float = 0.0
        self._prev_tokens_predicted: float = 0.0
        self._prev_prompt_tokens: float = 0.0
        self._prev_prompt_seconds: float = 0.0
        self._prev_tokens_predicted_seconds: float = 0.0

        if cfg.csv_out:
            self._csv_file = open(cfg.csv_out, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(["ts", "decode_tps", "prefill_tps", "n_output", "n_ctx_used"])

    @property
    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            return self._snapshot

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="llm-collector")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._csv_file:
            self._csv_file.close()

    # ── main loop ─────────────────────────────────────────────────────────────

    def _run(self) -> None:
        # Initial source detection
        source = "offline"
        prom_ok = self._try_init_prometheus()
        if prom_ok:
            source = "prometheus"
        else:
            log_path = self.cfg.resolve_log_file()
            if log_path:
                self._log_parser = LogParser(log_path)
                source = "log"

        # Fetch static model metadata from /models (retry until success)
        self._fetch_model_meta()

        interval = self.cfg.metrics_interval

        while not self._stop.is_set():
            loop_start = time.monotonic()
            sys_data = self._sys_poller.poll()

            if source == "prometheus":
                snap = self._poll_prometheus(sys_data)
                if snap is None:
                    # fallback to log if prometheus went away
                    source = "offline"
                    snap = self._build_offline_snapshot(sys_data)
            elif source == "log":
                snap = self._poll_log(sys_data)
            else:
                snap = self._build_offline_snapshot(sys_data)
                # retry prometheus detection every 10s
                if (time.monotonic() - self._started_at) % 10 < interval:
                    if self._try_init_prometheus():
                        source = "prometheus"
                    elif not self._log_parser:
                        log_path = self.cfg.resolve_log_file()
                        if log_path:
                            self._log_parser = LogParser(log_path)
                            source = "log"

            # Re-fetch /models if not yet populated, or every 60s (catches model swaps)
            elapsed = time.monotonic() - self._started_at
            if not self._model_meta or (elapsed % 60) < interval:
                self._fetch_model_meta()

            # Stamp static model metadata onto every snapshot
            if self._model_meta:
                snap.model_name    = snap.model_name or self._model_meta.get("model_name", "")
                snap.n_ctx_total   = snap.n_ctx_total or self._model_meta.get("n_ctx", 0)
                snap.n_ctx_train   = self._model_meta.get("n_ctx_train", 0)
                snap.model_size_gb = self._model_meta.get("size_gb", 0.0)
                snap.model_params_b = self._model_meta.get("params_b", 0.0)

            snap.source = source
            snap.ts = time.time()
            snap.uptime_wall_s = time.monotonic() - self._started_at
            snap.decode_history = self._decode_hist.values()
            snap.decode_sparkline = self._decode_hist.sparkline(60)
            snap.log_tail = list(self._log_tail[-self.cfg.log_lines_max:])
            snap.request_count = self._request_counter

            with self._lock:
                self._snapshot = snap

            # maintain poll interval
            elapsed = time.monotonic() - loop_start
            remaining = interval - elapsed
            if remaining > 0:
                self._stop.wait(timeout=remaining)

    def _try_init_prometheus(self) -> bool:
        url = self.cfg.prometheus_url()
        try:
            self._prom_poller = PrometheusPoller(url)
            data = self._prom_poller.poll()
            return bool(data)
        except Exception:
            self._prom_poller = None
            return False

    def _fetch_model_meta(self) -> None:
        """Non-blocking best-effort fetch of /models metadata."""
        try:
            self._model_meta = self._models_poller.fetch()
        except Exception:
            pass  # server not ready yet; retried next tick

    # ── Prometheus mode ───────────────────────────────────────────────────────
    #
    # Actual metric names (from /metrics HELP lines):
    #   llamacpp:tokens_predicted_total          counter — cumulative generation tokens
    #   llamacpp:tokens_predicted_seconds_total  counter — cumulative generation time
    #   llamacpp:prompt_tokens_total             counter — cumulative prompt tokens
    #   llamacpp:prompt_seconds_total            counter — cumulative prompt time
    #   llamacpp:predicted_tokens_seconds        gauge   — all-time avg generation tok/s
    #   llamacpp:prompt_tokens_seconds           gauge   — all-time avg prompt tok/s
    #   llamacpp:requests_processing             gauge   — slots active now
    #   llamacpp:n_decode_total                  counter — total llama_decode() calls

    def _poll_prometheus(self, sys_data: dict) -> Optional[MetricsSnapshot]:
        try:
            data = self._prom_poller.poll()
        except Exception:
            return None

        snap = MetricsSnapshot()
        snap.connected = True
        now = time.monotonic()

        # ── Live decode tok/s — Δtokens_predicted / Δtime ────────────────────
        curr_predicted = data.get("llamacpp:tokens_predicted_total", 0.0)
        curr_pred_secs = data.get("llamacpp:tokens_predicted_seconds_total", 0.0)
        dt = now - self._prev_prom_ts if self._prev_prom_ts else 0.0

        live_tps: Optional[float] = None
        if dt > 0 and curr_predicted > self._prev_tokens_predicted:
            delta_tok = curr_predicted - self._prev_tokens_predicted
            live_tps = delta_tok / dt
        elif dt > 0 and curr_pred_secs > self._prev_tokens_predicted_seconds:
            # Alternative: Δtokens / Δtime-in-seconds (more accurate when available)
            delta_tok = curr_predicted - self._prev_tokens_predicted
            delta_secs = curr_pred_secs - self._prev_tokens_predicted_seconds
            if delta_secs > 0:
                live_tps = delta_tok / delta_secs

        # ── All-time average throughput (gauges updated at request completion) ─
        avg_decode_tps  = data.get("llamacpp:predicted_tokens_seconds") or None
        avg_prefill_tps = data.get("llamacpp:prompt_tokens_seconds") or None

        # Use live rate during generation; fall back to all-time avg when idle
        processing = data.get("llamacpp:requests_processing", 0)
        snap.status = "GENERATING" if processing > 0 else "IDLE"

        if live_tps and live_tps > 0:
            snap.decode_tps = live_tps
            self._decode_hist.push(live_tps)
        elif avg_decode_tps and avg_decode_tps > 0 and snap.status == "IDLE":
            snap.decode_tps = avg_decode_tps
            # Only push to history once per completed request (detect via counter change)
            if curr_predicted > self._prev_tokens_predicted:
                self._decode_hist.push(avg_decode_tps)

        # Prefill: all-time average gauge (only meaningful at request boundary)
        snap.prefill_tps = avg_prefill_tps if avg_prefill_tps and avg_prefill_tps > 0 else None

        snap.avg_short = self._decode_hist.rolling_avg(self.cfg.avg_short_s)
        snap.avg_mid   = self._decode_hist.rolling_avg(self.cfg.avg_mid_s)
        snap.avg_long  = self._decode_hist.rolling_avg(self.cfg.avg_long_s)
        snap.peak_tps  = self._decode_hist.peak

        # Cumulative token counts (useful for session totals)
        snap.n_prompt = int(curr_predicted)   # reuse field for total predicted
        snap.n_output = int(data.get("llamacpp:prompt_tokens_total", 0))

        # CSV
        if self._csv_writer and snap.decode_tps:
            self._csv_writer.writerow([time.time(), snap.decode_tps, snap.prefill_tps, None, None])

        # Save counter state for next poll
        self._prev_prom_ts = now
        self._prev_tokens_predicted = curr_predicted
        self._prev_tokens_predicted_seconds = curr_pred_secs
        curr_prompt = data.get("llamacpp:prompt_tokens_total", 0.0)
        curr_prompt_secs = data.get("llamacpp:prompt_seconds_total", 0.0)
        self._prev_prompt_tokens = curr_prompt
        self._prev_prompt_seconds = curr_prompt_secs

        self._apply_sys(snap, sys_data)
        return snap

    # ── Log mode ─────────────────────────────────────────────────────────────

    def _poll_log(self, sys_data: dict) -> MetricsSnapshot:
        snap = MetricsSnapshot()
        snap.connected = True

        new_lines = self._log_parser.poll()
        if new_lines:
            self._log_tail.extend(new_lines)
            if len(self._log_tail) > self.cfg.log_lines_max:
                self._log_tail = self._log_tail[-self.cfg.log_lines_max:]

        # Absorb any completed request record
        rec = self._log_parser.last_record
        if rec and (not self._req_hist.latest() or
                    rec.ts > (self._req_hist.latest().ts if self._req_hist.latest() else 0)):
            self._req_hist.push(rec)
            self._request_counter += 1
            self._log_parser.last_record = None
            if rec.decode_tps:
                self._decode_hist.push(rec.decode_tps)
            if self._csv_writer:
                self._csv_writer.writerow([
                    rec.ts, rec.decode_tps, rec.prefill_tps,
                    rec.n_output, rec.n_ctx,
                ])

        snap.model_name  = self._log_parser.model_name
        snap.n_ctx_total = self._log_parser._ctx_total
        snap.status      = "GENERATING" if self._log_parser.is_generating else "IDLE"
        snap.task_id     = self._log_parser.current_task

        # ── Live decode rate from slot print_timing (new format) ─────────────
        # These lines fire every ~3s during generation — much better than
        # waiting for end-of-request.
        if self._log_parser.live_tps is not None and self._log_parser.is_generating:
            snap.decode_tps  = self._log_parser.live_tps
            snap.server_tg3s = self._log_parser.live_tg3s
            snap.prefill_tps = self._log_parser.live_prefill_tps
            snap.n_output    = self._log_parser.live_n_decoded
            # n_ctx_used = at least n_decoded; prompt tokens not in this log format
            snap.n_ctx_used  = self._log_parser.live_n_decoded
            # Push live sample into rolling history (dedupe: only if changed)
            last_hist = self._decode_hist.latest()
            if last_hist is None or abs(snap.decode_tps - last_hist) > 0.01:
                self._decode_hist.push(snap.decode_tps)
        else:
            # Fall back to last completed request values when idle
            last = self._req_hist.latest()
            if last:
                snap.decode_tps  = last.decode_tps
                snap.prefill_tps = last.prefill_tps
                snap.n_prompt    = last.n_prompt
                snap.n_output    = last.n_output
                snap.n_ctx_used  = last.n_ctx
            snap.last_request = last

        snap.last_request = self._req_hist.latest()
        snap.avg_short = self._decode_hist.rolling_avg(self.cfg.avg_short_s)
        snap.avg_mid   = self._decode_hist.rolling_avg(self.cfg.avg_mid_s)
        snap.avg_long  = self._decode_hist.rolling_avg(self.cfg.avg_long_s)
        snap.peak_tps  = self._decode_hist.peak

        self._apply_sys(snap, sys_data)
        return snap

    # ── Offline snapshot ──────────────────────────────────────────────────────

    def _build_offline_snapshot(self, sys_data: dict) -> MetricsSnapshot:
        snap = MetricsSnapshot()
        snap.connected = False
        snap.status = "OFFLINE"
        snap.avg_short = self._decode_hist.rolling_avg(self.cfg.avg_short_s)
        snap.avg_mid   = self._decode_hist.rolling_avg(self.cfg.avg_mid_s)
        snap.avg_long  = self._decode_hist.rolling_avg(self.cfg.avg_long_s)
        snap.peak_tps  = self._decode_hist.peak
        snap.last_request = self._req_hist.latest()
        self._apply_sys(snap, sys_data)
        return snap

    @staticmethod
    def _apply_sys(snap: MetricsSnapshot, sys_data: dict) -> None:
        snap.cpu_pct       = sys_data.get("cpu_pct", 0.0)
        snap.cpu_per_core  = sys_data.get("cpu_per_core", [])
        snap.load_avg      = sys_data.get("load_avg", 0.0)
        snap.ram_used_gb   = sys_data.get("ram_used_gb", 0.0)
        snap.ram_total_gb  = sys_data.get("ram_total_gb", 0.0)
        snap.model_rss_gb  = sys_data.get("model_rss_gb", 0.0)

    # ── User actions ──────────────────────────────────────────────────────────

    def reset_stats(self) -> None:
        self._decode_hist.reset()

    def clear_history(self) -> None:
        self._req_hist.reset()
        self._decode_hist.reset()
        self._log_tail.clear()
