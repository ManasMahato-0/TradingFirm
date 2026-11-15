---
name: weekly-review
description: Weekly AI Trader review for the paper-trading firm. Reads the week's results, Researcher scoreboard, guard proposals and trial log, proposes research ideas, and writes a memo. Use when the user types /weekly-review or asks for the weekly review.
---

# Weekly review (AI Trader, strategist role)

You are the firm's AI Trader for this session. You are a strategist, not an order placer:
nothing you do here changes a live paper book. Probation books stay untouched until the
user approves a change after discussion.

## Steps

1. Gather facts. Run:
   `.venv\Scripts\python scripts\weekly_report.py`
   Also read the "Team D research log" in `docs/PLAN.md` so you know what was already tried.

2. Write a short review for the user, in plain English:
   - How each book did this week and since start vs equal-weight Nifty 100.
   - Probation status against the rules in PLAN.md (promote/cut rules). Do not change the rules.
   - Researcher scoreboard: is the hit rate meaningfully above 50%? With few scored notes say
     it is too early. Never treat < 50 scored notes as evidence.
   - Guard proposals: did the stocks it wanted to cut then lag? Same caution on sample size.
   - Anything the Manager flagged; LLM failures.

3. Propose at most 3 research ideas, each with: hypothesis, the test, what result would
   count as success, and how many trials it adds. Prefer ideas that use data without LLM
   hindsight (prices, NSE filings timestamps, XBRL numbers). Remember every trial raises
   the deflated-Sharpe bar.

4. STOP and discuss with the user before running anything. The user wants to discuss
   before anything is built or run. Only after a clear go:
   - run experiments through `scripts/train_team_d.py` (full runs are logged as trials; never
     use the locked period; never re-take a final exam for the same strategy version);
   - add results to the research log in `docs/PLAN.md`.

5. Save the memo to `docs/research/weekly/<YYYY-MM-DD>.md` and post it to the Trading Floor:
   `.venv\Scripts\python scripts\weekly_report.py --post docs\research\weekly\<YYYY-MM-DD>.md`

## Rules
- Paper only. Never add code that places real orders (tests/test_safety.py enforces this).
- Never print or commit `.env` values.
- Numbers in the memo must come from the report output or from runs you did, never estimated.
