# Trading Firm

A paper-trading lab for Indian markets (NSE), built to answer one question honestly: **does any trading idea survive real costs and a fair test?**

Everything here is **paper money**. There is no real-order code anywhere in the project, and a test (`tests/test_safety.py`) fails if any ever appears.

> Not financial advice. This is a research project. Past results, including the ones recorded here, say nothing about the future.

## What it does

- Saves market data every day: prices (1-minute and daily), NSE company filings, board-meeting dates, quarterly results, news, option prices.
- Backtests ideas with walk-forward testing, real Indian trading costs, and a locked final-exam period that each idea gets to use exactly once.
- Runs live paper accounts on a schedule: a swing-trading experiment and four intraday "practice" accounts.
- Uses local AI helpers (Ollama + FinBERT) to read news and filings and write plain-English notes. **The AI never places orders.**
- Shows everything in a local dashboard with a Home page that tells you what to do, a live activity feed, and a Settings page with every switch.

## Where things stand

| Area | Status |
|---|---|
| Swing stocks | Live paper test starts 30 Nov 2026 (momentum + LightGBM picker), with plain momentum as a yardstick |
| Intraday practice accounts | Four paper accounts (trained on history, learns live, random, cash) launch 23 Nov 2026 |
| Intraday stocks | Parked: the best research idea failed its final exam |
| Option teams | Parked: intraday has no price history; overnight baselines all failed |
| Fundamentals (results numbers) | Tested, no predictive edge found |
| Head (capital allocator) | Built in advice-only mode: writes a daily firm report and a proposal, moves no money. Currently says HOLD |

Nothing has passed a locked-period exam so far. The full research log, including every failure, is in [`docs/PLAN.md`](docs/PLAN.md).

## How it stays honest

- **Pre-registration.** Rules, thresholds and pass criteria are written into `docs/PLAN.md` before a test runs. Changing one afterwards needs a dated, disclosed amendment.
- **Locked final exams.** The most recent months of data are never seen during research. Each strategy version gets one try.
- **Every trial is counted.** More tries raise the bar (deflated Sharpe).
- **Fair benchmarks.** Ideas must beat cash, a simple momentum rule, or equal-weight Nifty 100, depending on the test.
- **Backtest = live.** The same step code runs both, and a test proves a replayed day equals a backtest of that day.
- **AI is advisory.** Notes are checked so every number comes from the data; the risk watcher can only observe.

## Layout

```
firm/        broker, risk engine, costs, backtester, features, strategies,
             live runner, sandbox, AI agents, control board
scripts/     recorder, runner, sandbox service, research and exam scripts
dashboard/   Streamlit app (Home, Live feed, Money, Notes, Settings, Research, System)
tests/       about 180 tests
docs/        PLAN.md: design, rules, and the full research log
```

## Setup (Windows)

Built and tested on Windows 11 with Python 3.12.

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
```

1. Fill in `.env` with your Fyers app keys (free Fyers API account). Never commit `.env`; it is git-ignored.
2. Install [Ollama](https://ollama.com) and run `ollama pull qwen3:8b` for the AI helpers (optional; they pause if it's missing).
3. Start the background programs. These also start at Windows login if you add the startup entries:

```bat
.venv\Scripts\python scripts\recorder.py run
.venv\Scripts\python scripts\recorder.py install-startup
.venv\Scripts\python scripts\sandbox.py live
.venv\Scripts\python scripts\sandbox.py install-startup
```

4. Open the dashboard (it only listens on your own computer):

```bat
.venv\Scripts\streamlit run dashboard\app.py
```

then go to <http://localhost:8501>.

## Daily use

Fyers logins expire every day. On a trading morning, open the dashboard's **Home** page and follow the three steps to log in (or run `scripts\fyers_login.py`). You get a Windows notification at 08:45 if you have not logged in. The system does the rest by itself, including a daily backup of the data. All switches, start/stop buttons and the emergency halt are on the **Settings** page.

## Useful commands

```bat
.venv\Scripts\python scripts\recorder.py status                       # data collector health
.venv\Scripts\python scripts\runner.py dry-run --start 2026-06-01     # rehearse the swing runner
.venv\Scripts\python scripts\sandbox.py replay --day 2026-11-12       # replay a recorded day
.venv\Scripts\python scripts\research_intraday.py --quick             # smoke-test the research scripts
.venv\Scripts\python -m pytest -q                                     # all tests (about 10 minutes)
```

## Data sources and limits

Everything is free: Fyers (free plan: about one year of intraday history, rate-limited), Yahoo Finance for long daily history, and NSE's public website feeds and archives, which are unofficial and can change without notice. The collectors are deliberately polite (throttled, one request at a time where possible).

## License

None yet. All rights reserved by the author until one is added.
