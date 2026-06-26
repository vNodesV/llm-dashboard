#!/usr/bin/env python3
"""
llm-watch — real-time llama.cpp monitor
htop/btop for your local LLM.

Usage:
  llm-watch.py [--host HOST] [--port PORT] [--log LOG_FILE]
               [--interval HZ] [--no-color] [--csv FILE]

Source priority:
  1. Prometheus /metrics (auto-detected on startup)
  2. Log file tail (--log or auto-detected common paths)
  3. System metrics only (psutil, always active)

Keys:
  q   Quit
  r   Reset rolling stats
  c   Clear history
  l   Toggle log view
"""

import argparse
import curses
import sys
import os

# ── Dependency check (friendly error before curses init) ─────────────────────
try:
    import psutil  # noqa: F401
except ImportError:
    print("Error: psutil is required. Install with: pip install psutil", file=sys.stderr)
    sys.exit(1)

from config import Config
from metrics import Collector
from ui import Dashboard


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="llm-watch",
        description="Real-time llama.cpp monitor — htop for your LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-connect to localhost:8080 (default llama-server port)
  ./llm-watch.py

  # Remote server
  ./llm-watch.py --host 192.168.1.50 --port 8080

  # Force log parsing (no Prometheus)
  ./llm-watch.py --log /tmp/llama.log --no-prometheus

  # Record tok/s history to CSV
  ./llm-watch.py --csv session.csv

  # 4 refreshes/sec
  ./llm-watch.py --interval 4
        """,
    )
    p.add_argument("--host",        default="127.0.0.1",  help="llama-server host (default: 127.0.0.1)")
    p.add_argument("--port",        default=8080, type=int, help="llama-server port (default: 8080)")
    p.add_argument("--log",         default=None,          help="Path to llama-server log file")
    p.add_argument("--interval",    default=2.0, type=float, help="UI refresh rate in Hz (default: 2)")
    p.add_argument("--no-color",    action="store_true",   help="Disable colors")
    p.add_argument("--no-prometheus", action="store_true", help="Skip Prometheus /metrics; use log only")
    p.add_argument("--csv",         default=None,          help="Record session metrics to CSV file")
    p.add_argument("--history",     default=120, type=int, help="Sparkline history depth (default: 120)")
    p.add_argument("--thresh-warn", default=10.0, type=float, help="tok/s yellow threshold (default: 10)")
    p.add_argument("--thresh-good", default=20.0, type=float, help="tok/s green threshold (default: 20)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        host=args.host,
        port=args.port,
        log_file=args.log,
        refresh_hz=args.interval,
        metrics_poll_hz=max(args.interval * 2, 4.0),
        color=not args.no_color,
        prefer_prometheus=not args.no_prometheus,
        csv_out=args.csv,
        history_len=args.history,
        thresh_warn=args.thresh_warn,
        thresh_good=args.thresh_good,
    )

    collector = Collector(cfg)
    collector.start()

    dashboard = Dashboard(collector, cfg)

    try:
        curses.wrapper(dashboard.run)
    except KeyboardInterrupt:
        pass
    finally:
        collector.stop()


if __name__ == "__main__":
    main()
