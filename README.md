# llm-watch

**htop/btop for llama.cpp** — real-time curses dashboard for llama-server.

```
┌─ llama.cpp Monitor ─────────────────────────── 11:48:02 ─
│
│  STATUS  ● GENERATING        MODEL  DeepSeek-Coder-V2-Lite
│  REQUEST #482                UPTIME 2d 14h 33m
│
│  Decode    22.81 tok/s  ████████████████████████░░░░░░░░░
│  3s Avg    22.44
│  1m Avg    21.98
│  5m Avg    21.71
│  Peak      24.71
│
│  Prefill   1864 tok/s
│  Prompt    7,421 tokens
│  Output      911 tokens
│  Context  8,332 / 32,768
│
├─ System ─────────────────────────────────────────────────
│  CPU     97.4%
│  Load    31.2
│  RAM     72.4 / 128.0 GB
│  RSS     38.6 GB
│
├─ Cores ──────────────────────────────────────────────────
│   97% ████████████████████████████████████████
│   98% █████████████████████████████████████████
│   96% ███████████████████████████████████████▊
│   99% ████████████████████████████████████████
│
├─ Decode History ─────────────────────────────────────────
│  ▁▂▂▃▄▅▆▆▇██▇▆▅▄▃▂▂▃▄▅▆▇█▇▆▅▄▃▄▅▆▇██▇▆▅
│
├─ Last Request ───────────────────────────────────────────
│  Prefill  1,844 tok/s
│  Decode    22.81 tok/s
│  Tokens      911
│  Duration  41.7 s
│
├─ ⬡ prometheus  http://127.0.0.1:8080 ───────────────────
  q Quit    r Reset Stats    c Clear History    l Log View
```

## Install

```bash
git clone https://github.com/vNodesV/llm-dashboard
cd llm-dashboard
make install    # creates .venv + installs psutil
```

Single dependency: `psutil`. Everything else is stdlib.

## Run

```bash
# Auto-detect: tries Prometheus first, falls back to log file
make run

# Or directly:
.venv/bin/python llm-watch.py

# Remote llama-server
.venv/bin/python llm-watch.py --host 192.168.1.50 --port 8080

# Force log file (skip Prometheus)
.venv/bin/python llm-watch.py --log /tmp/llama.log --no-prometheus

# Record to CSV
.venv/bin/python llm-watch.py --csv session.csv

# System Python (no venv)
pip install psutil
python3 llm-watch.py
```

## Data sources

| Source | How | When used |
|--------|-----|-----------|
| **Prometheus `/metrics`** | HTTP poll every 250ms | Auto-selected if llama-server has `--metrics` flag |
| **Log file tail** | Keep-open seek, parse new lines only | Fallback if no /metrics |
| **psutil** | CPU/RAM/per-core | Always active |

Auto-detection order: Prometheus → common log paths → system-only.

Common auto-detected log paths:
- `/tmp/llama.log`
- `/tmp/llama-server.log`
- `~/llama-server.log`

## Keys

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset rolling averages + peak |
| `c` | Clear all history |
| `l` | Toggle raw log view |
| `ESC` | Also quits |

## Options

```
--host HOST          llama-server host (default: 127.0.0.1)
--port PORT          llama-server port (default: 8080)
--log FILE           Log file path (overrides auto-detect)
--interval HZ        UI refresh rate (default: 2.0 Hz)
--no-color           Disable colors
--no-prometheus      Skip /metrics; use log only
--csv FILE           Record tok/s history to CSV
--history N          Sparkline depth in samples (default: 120)
--thresh-warn N      tok/s yellow threshold (default: 10)
--thresh-good N      tok/s green threshold (default: 20)
```

## Enabling Prometheus in llama-server

```bash
llama-server --metrics --port 8080 -m model.gguf [...]
```

With Prometheus, tok/s updates continuously during generation (not just at completion).

## Architecture

```
llm-watch.py     entry point, arg parsing
config.py        Config dataclass — all tunable settings
metrics.py       Collector (background thread)
  ├── PrometheusPoller  — HTTP /metrics poll
  ├── LogParser         — incremental log tail
  └── SystemPoller      — psutil wrapper
history.py       RollingHistory, sparklines, percentiles
ui.py            curses Dashboard, resize handling
```

The Collector runs in a daemon thread; the UI reads an atomically-swapped snapshot on each tick. No locks held during rendering.

## Log format support

Parses standard `llama_print_timings` output (all llama.cpp versions):
```
llama_print_timings:        eval time = 39876.54 ms /   910 runs   (  43.82 ms per token,    22.82 tokens per second)
llama_print_timings: prompt eval time =   456.78 ms /  7421 tokens (   0.06 ms per token,  1864.32 tokens per second)
llama_print_timings:       total time = 40456.78 ms /  8331 tokens
```

Also parses newer JSON timings format (llama-server b3xxx+).
