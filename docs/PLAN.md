# AI Trading Firm: Plan

Paper trading only on NSE. No real orders until explicitly decided later.

## Organisation

```
                         HEAD (LLM)
          compares teams, moves capital monthly, firm report
                              │
                   SUPERVISOR / RISK ENGINE (code)
            firm-wide limits, kill switch, veto on every order
                              │
   ┌──────────────┬───────────┴──┬────────────────┬────────────────┐
 Swing Stocks   Intraday Stocks  Intraday Options  Overnight Options
   ₹1L            ₹1L              ₹1L               ₹1L
```

Each team:

| Role | What it is | What it does |
|---|---|---|
| AI Trader | LLM | Strategist. Forms hypotheses, designs and trains RL policies, reviews results, allocates capital across its policies. During market hours can only reduce risk (cut size, skip a stock, pause a policy). Never places orders. |
| RL policies | Code (PPO etc.) | Make the actual buy/sell decisions every bar. |
| Researcher | LLM | Reads news, filings, results, earnings calls. Writes per-stock notes and features. |
| Manager | LLM | Reads daily performance, flags drift, writes plain-English daily note. |

Every order, from any team, passes through the risk engine, then the paper broker.

### Why LLMs don't trade directly

- LLMs know market history from training data, so their backtests use hindsight and cannot be trusted.
- Too slow and inconsistent for per-bar decisions.
- RL and statistics handle price series better; LLMs handle text and research better.

## Anti-overfitting rules

- Locked final test period (most recent ~6 months) never shown during research.
- Every strategy trial is counted; more trials means a higher bar (deflated Sharpe).
- Must beat dumb baselines: buy-and-hold Nifty, simple moving-average rule.
- Walk-forward testing: train on the past, test on unseen later periods, repeat.
- A strategy gets capital only after several weeks of live paper results.

## Data (free)

| Source | Use | Limits |
|---|---|---|
| Fyers API | Recent daily, intraday, live quotes, option chain | Daily 5 years, intraday 1 year, 5 requests/second and 50/minute, token expires daily |
| Yahoo Finance | Long daily history for swing backtests | Some missing days |
| NSE/BSE announcements | Results, filings, insider trades, corporate actions | No official API, poll politely |
| News RSS (ET, Moneycontrol, BS, Mint, Google News) | Headlines | Recent only |
| GDELT | Historical news back to 2015 | Patchy for India |

We record intraday bars, option chain and news ourselves every day, because free sources don't keep history.

News timestamps are the time we first saw the item. Backtests may only use news known at that moment.

## LLMs (₹0/month)

| Job | Model |
|---|---|
| Headline sentiment, first filter | FinBERT (local) |
| Tagging, event type, summaries, daily check-ins | Qwen/Llama 7B via Ollama (local GPU) |
| Earnings calls, results deep-dives | Gemini free tier |
| Weekly strategy research, big decisions | Claude Pro sessions (human in the loop) |

LLM backends are swappable via config.

## Build order

1. [x] Foundation: costs, paper broker, risk engine, storage, Fyers + Yahoo data
2. [x] Recorders: daily intraday bars, option chain, news and filings (`scripts/recorder.py`)
3. [x] Backtester with walk-forward and locked test period (`firm/backtest/`, `scripts/backtest_baselines.py`)
4. [ ] Swing Stocks team: baselines, RL environment, PPO, AI Trader research loop
5. [x] Live paper runner (`scripts/runner.py`, run by the recorder after 16:00) + Dashboard v1
   (`dashboard/app.py`: Trading Floor, Analytics, Research, System)
6. [x] LLM layer (`firm/llm`, `firm/agents`): Ollama qwen3:8b + FinBERT. Researcher (notes on
   filings and headlines every 5 min, scored forward), Manager (daily note, number-checked),
   AI Trader guard (risk-reducing proposals, shadow mode), `/weekly-review` in Claude Code
7. [x] Intraday Stocks team: researched, no strategy passed the final exam, parked in cash (see log)
8. [x] Options teams: Intraday Options parked (no price history; recorder collects chains), Overnight Options researched on real NSE data, all baselines failed, parked (see log)
9. [ ] Head, scheduler running all day

## Baseline results (swing research period 2008 to May 2026)

| | Nifty 50 hold | Equal-weight N100* | Momentum 12-1 | Trend SMA 50/200 |
|---|---|---|---|---|
| CAGR | 7.2% | 21.4% | 23.8% | 15.5% |
| Sharpe | 0.12 | 0.77 | 0.77 | 0.48 |
| Max drawdown | -59.9% | -55.7% | -63.5% | -58.3% |

\* Survivorship bias: today's Nifty 100 members held since 2008. The fair bar for any
strategy on this universe is equal-weight N100, not Nifty 50.

## Team D research log

**2026-11-14, walk-forward 2011 to May 2026 (yearly retrain):** no AI picker passed.
LightGBM (Sharpe 0.23) chased volatile names and over-traded; PPO picker (Sharpe -0.64)
learned to sit in cash (reward on excess-vs-average minus costs makes "hold nothing" safe).
Nifty-only rule dial blind to the 2018-19 midcap crash. Momentum passed research
(Sharpe 0.94 vs EW 0.86, DD -41%, beat EW 62% of years).

**2026-11-14, momentum final exam (May to Nov 2026): FAILED.** Return -10.9% vs
equal-weight +5.3%. Version f6f6f013d582 cannot retake. Lesson: survivorship-inflated
research looked strong; the locked period caught it.

Next fixes: AI picker refines momentum's top-20 candidates; 4-week horizon with monthly
rebalance; always invested by rank (dial alone controls cash); dial sees breadth and own
drawdown; idle cash earns liquid-fund rate; then results-calendar features.

**2026-11-14, fixes built:** `MomentumPlus` (momentum top-20 shortlist, AI picker
chooses 10, monthly, always invested by rank), symmetric PPO reward, dial sees the
broad equal-weight index and own drawdown, idle cash earns 6.5%/yr in backtests,
strategy versions hash their feature code too. Walk-forward running.

To do with the live paper runner: shadow-track momentum (no capital) as a yardstick.

**2026-11-14, MomentumPlus walk-forward 2011 to May 2026:**

| Config | Sharpe | Max DD | Years beating EW |
|---|---|---|---|
| Equal-weight N100 | 0.86 | -37.6% | - |
| momplus none/none (momentum order) | 1.07 | -40.3% | 81% |
| momplus none/ddrule | 1.03 | -24.5% | 62% |
| momplus lgbm/none | 1.03 | -38.5% | 69% |
| momplus lgbm/none + results | **1.18** | -37.1% | 75% |
| momplus lgbm/ddrule + results | 1.13 | **-22.0%** | 50% |
| momplus ppo (all dials) | 0.31 to 0.73 | | 31 to 44% |

Findings: holding buffer + month-end + cash yield improved plain momentum; drawdown rule
cuts max DD from ~40% to ~23% for a small return cost; PPO picker and PPO dial still lose;
results features help LightGBM. Signal is post-results drift: stocks with the strongest
recent results reactions beat the average by +0.37%/month, the weakest lag by -0.47%.
No pre-results run-up visible in a simple sort. Deflated Sharpe fails for all after 16
trials (luck bar ~1.1 to 2.3 annual Sharpe; broken early PPO trials inflate it).

Note: the May-Nov 2026 window has now been seen once. Later exams on it are weaker
evidence, so forward paper trading is the real test.

**2026-11-14, decision: probation tier (option B).** Strategies passing 3 of 4 research
gates may paper-trade with reduced capital; forward paper results decide promotion.
Candidate: `momplus_lgbm_none+res`. Plain momentum shadow-tracked with no capital.

**2026-11-14, live runner built.** Books open 30 Nov 2026 (first decision at that close,
first fills 1 Dec open), probation to 31 May 2027. Probation runs on the full Rs 1L, not
Rs 25k: at Rs 25k a slot is Rs 2,450 and 32 of 100 stocks cost more than that per share,
so the live book would differ from the researched one. Promote: return >= equal-weight
- 5pts, max DD better than -25%, no risk-engine blocks, trades match a re-run backtest.
Cut: DD past -25% or trailing equal-weight by 10pts.
Dry-run rehearsal over 29 May to 13 Nov 2026 (the already-seen exam window): probation
-9.7%, shadow momentum -11.6%, equal-weight +2.5%. Not a new trial; flagged as a warning.

## Intraday Stocks research log

**2026-11-15, baselines walk-forward 17 Nov 2025 to 14 Aug 2026** (5-minute bars, 50 most
traded stocks picked from the first 3 months, 1-month tests, Rs 1L, flat by 15:15, round
trip cost ~0.17% incl. slippage). 6 trials logged.

| Strategy | Before costs | After costs | Sharpe after | Round trips/day |
|---|---|---|---|---|
| Opening range breakout | +17.8% | -28.1% | -3.13 | 5.5 |
| Nifty first-half-hour momentum | +9.9% | -11.2% | -6.15 | 2.1 |
| Gap fade | +17.4% | -2.7% | -0.86 | 1.9 |
| Gap follow | +0.2% | -17.1% | -2.21 | 1.9 |
| VWAP reversion | -5.3% | -52.2% | -7.51 | 8.0 |
| Filing reaction | +2.3% | +0.2% | -1.72 | 0.2 |

None pass. Some gross edges (ORB, gap fade, Nifty momentum, Sharpe ~0.9-1.1 before costs)
but costs take all of it; 9 months is too short for those gross Sharpes to be significant.

**2026-11-15, PRE-REGISTERED before any run: two filtered intraday variants.**
Settings are fixed here, before any number is seen. Exactly two new trials, same data,
same walk-forward (1-month tests, 17 Nov 2025 to 14 Aug 2026), same costs, no final exam.
1. `gap_fade_big`: `Gap(mode="fade", min_gap=0.03)`. The gap is the stock's open vs its
   previous close MINUS Nifty's gap (relative gap), as in the baseline. Nothing else changes.
2. `orb_filtered`: ORB as in the baseline (`vol_mult=1.5`, slot 0.24) plus (a) market
   filter: long only if Nifty's close at the decision bar is above its previous close,
   short only if below; (b) at most ONE new entry per day across all stocks.
Promotion to a shadow paper book needs ALL of: Sharpe after costs > 0.5; max drawdown
better than -15%; at least 60% of months beating cash; and at least 30 round trips
(fewer is "too few trades to judge" = not promoted). Deflated Sharpe is reported but
excluded (it cannot pass at 8+ trials), same as the swing probation precedent.
If neither passes: team parked in cash; both filtered variants go to SHADOW (no capital)
once a live intraday runner exists. That choice is made now, not after the results.
Code changes for this must default to old behaviour so the six baselines still reproduce.

**2026-11-15, RESULT of the two pre-registered variants** (same window and costs; 2 more
trials logged, 8 intraday trials total; no final exam taken; settings unchanged from the
pre-registration; a --quick crash check ran first on Nov-Feb and changed nothing).

| | gap_fade_big | orb_filtered |
|---|---|---|
| Round trips (total, 9 months) | 78 | 159 |
| Yearly before costs / Sharpe | +18.6% / 1.68 | +13.4% / 1.27 |
| Yearly after costs / Sharpe | **+13.0% / 0.95** | +2.7% / -0.74 |
| Max drawdown | -2.0% | -3.5% |
| Months beating cash | 62% (5 of 8) | 38% |
| Costs paid (Rs) | 1,922 | 3,834 |
| Deflated Sharpe (excluded) | 0.02 | 0.00 |
| Pre-registered verdict | **PROMOTE** (shadow book) | no |

Caveats: gap_fade_big passes by thin margins (months beating cash is 5 of 8: one month
flipping fails it); 78 trades in one market regime; the "big gaps only" idea came after
seeing that baseline gap fade had a before-cost edge but too many trades, so it is
informed by earlier results even though the settings were fixed before this run; the
13.0% includes ~6.5% idle-cash interest (edge above cash is roughly 6.5%/yr); deflated
Sharpe fails as expected. ORB's market filter did not help; ORB stays dead.
Decision per pre-registration: gap_fade_big becomes the intraday candidate for a shadow
paper book (no capital) once the live intraday runner exists; orb_filtered is shadow-only
as pre-announced. The locked intraday window (14 Aug to 13 Nov 2026, never seen) is still
unused: a final exam for gap_fade_big is one shot and is the next real test.

**2026-11-15, FINAL EXAM CRITERIA for gap_fade_big (fixed before the run).** One shot on the
locked intraday window 14 Aug 2026 to the last recorded day, same settings, same costs.
Passes only if ALL hold:
1. net return over the window beats cash (6.5%/yr pro rata): the trading itself must pay
2. max drawdown no worse than -5% (research worst was -2.0%)
3. round trips per day at most 2x research (research 0.48/day, so at most 0.96/day)
4. at least 10 round trips; with fewer the result is INCONCLUSIVE (exam still consumed)
Sharpe is reported but not a gate (3 months is too short). Failed or inconclusive:
gap_fade_big is parked and never re-examined on this version.

**2026-11-15, FINAL EXAM RESULT, gap_fade_big (version 7056da374873): FAILED.** Locked window
14 Aug to 13 Nov 2026 (63 trading days, never seen in research): return +0.27% vs cash
+1.64% (FAIL); max drawdown -1.67% (pass); 19 round trips, 0.30/day (pass); Sharpe -1.19;
win rate 47%; costs Rs 468. Idle-cash interest alone is ~1.6%, so trading itself lost
about 1.4% after costs. Version cannot retake. Decision: Intraday Stocks team PARKED in
cash; no strategy passed; gap_fade_big gets no shadow book. Lesson (second time after
swing momentum): a thin research pass, even with pre-registered settings, did not survive
the locked period; the exam gate did its job. The live intraday runner is not justified
now. Revisit only with new ideas, a longer history (data keeps recording), or better costs.

**2026-11-15, FUNDAMENTALS (XBRL results numbers): PRE-REGISTERED step-2 rule, fixed before any
look at the data.** Source: NSE corporates-financial-results feed + XBRL files (exist from
~mid-2018 only; first YoY growth ~late 2019). Public time = first filing's broadcast time;
a quarter's numbers are used only from the next trading day. Step 2 is DESCRIPTIVE (not a
logged trial): at each month-end decision from Oct 2019 to the swing lock (14 May 2026),
sort the 100 stocks into 5 groups by each of 5 measures and compare next-month return
(open-to-open) minus the equal-weight average. The 5 measures: revenue growth YoY, net
profit growth YoY, EPS growth YoY, net profit margin change YoY, profit-growth surprise
(this quarter's YoY profit growth vs the company's own previous 8 quarters of it).
A measure "passes" if ALL: (1) top group minus bottom group >= +0.30%/month; (2) that
spread is positive in BOTH halves of the sample; (3) rank correlation of the 5 group
means with group number >= 0.7. PROCEED to a model step only if at least 2 of the 5
measures pass (guards against picking a lucky one of five). Otherwise stop; nothing is lost.
If it proceeds: variant with these features, walk-forward ~2022 to lock with 2.5 years of
training, 2-3 counted trials; a winner gets its own shadow book, never swapped into the
probation book. Also independent of research: results numbers are posted to the Trading
Floor and given to the Researcher (code-extracted, no LLM, no hindsight).

**2026-11-15, FUNDAMENTALS STEP-2 RESULT: 0 of 5 measures passed, so STOP (no model).**
Data: 3,175 quarters for all 100 stocks (87% consolidated), NSE old feed to Dec 2024 +
integrated-filing feed from Apr 2025. 78 monthly decisions, Oct 2019 to Mar 2026, ~75-85
stocks per month. Next-month return vs equal-weight, % per month, groups 1 (lowest) to 5:

| Measure | G1 | G2 | G3 | G4 | G5 | Top-bottom | 1st half | 2nd half | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| Revenue growth YoY | 0.34 | 0.06 | -0.17 | 0.10 | 0.30 | -0.04 | -0.99 | +0.91 | fail |
| Profit growth YoY | 0.50 | 0.29 | -0.19 | -0.27 | 0.43 | -0.07 | -0.69 | +0.56 | fail |
| EPS growth YoY | 0.22 | 0.06 | -0.29 | 0.16 | 0.49 | +0.27 | +0.10 | +0.44 | fail |
| Net margin change | 0.39 | 0.12 | -0.26 | -0.02 | 0.41 | +0.02 | -0.54 | +0.59 | fail |
| Profit surprise vs own history | -0.02 | -0.38 | 0.15 | -0.14 | 0.62 | +0.64 | +0.21 | +1.06 | fail (not monotone, t=1.3) |

Closest was profit surprise (+0.64%/month, positive in both halves, but t = 1.3 and not
monotone). The raw numbers add little beyond what the price reaction already captured.
An unregistered observation (NOT acted on, would be a fresh look): both extreme groups beat
the middle (U-shape). Fundamentals as a picker feature are shelved. What stays: the data
pipeline, the live recorder job (checks NSE every 30 min), the Trading Floor result lines
and the Desk "Results numbers" tab, which need no research claim.

## Intraday sandbox (live paper forward test): PRE-REGISTERED 2026-11-15, before any training or sanity run

Purpose: a cheap live test of whether a model trained on history, or a blank learner, finds
anything in 5-minute intraday stock data. Not a proven team; no capital is ever promoted from it
without a real period of results. Cap: exactly these four accounts, Rs 1L paper each.

**Common to all four:** the 50 liquid stocks already chosen (first 3 months of data only);
5-minute bars built from 1-minute bars with `resample_5m`; same cost model, same risk limits
(intraday_stocks); long and short; max 4 positions of 24% each; flat by 15:15; one entry per
stock per day; decisions at a bar's close for bars starting 09:45 to 14:25; ledger fills at the
NEXT bar's open (same as the backtest `Session.step`); idle cash earns 6.5%/yr in all four.
Launch: Monday 23 Nov 2026 (Monday 16 Nov may be used as an uncounted shake-down). If the live
feed is not verified by then, launch is the first Monday after it is. Days are counted from launch.

**A, trained on history:** LightGBM regressor, features (14, all causal at the bar close): returns
over the last 1, 3, 6, 12 bars minus Nifty's; VWAP deviation (%) and VWAP deviation in units of
recent bar volatility; day return minus Nifty's; opening gap minus Nifty's gap; position inside the
opening range ((close - range middle) / range width, clipped +/-5); log volume ratio vs the same
time of day over 20 days; recent bar volatility; time of day; Nifty's day return; Nifty's 6-bar
return. Label: stock's return from the open of the next bar to the open 7 bars later (6-bar hold)
minus Nifty's return over the same window; samples never cross the day boundary. Params:
n_estimators 300, learning_rate 0.03, num_leaves 15, min_child_samples 1000, subsample 0.7
(bagging every iteration), colsample_bytree 0.8, seed 0, label clipped +/-5%, every decision bar
used. Entry: predicted 6-bar excess return >= +0.25% goes long, <= -0.25% goes short; at most 2
new entries per bar; exit after 6 bars. Initial training: all recorded data before launch;
retrained on the first trading day of each month on all data to the previous day.
ZERO TRADES IS A VALID RESULT: the threshold is never loosened after seeing a sanity run.

**B, blank learner:** identical model, rule and parameters to A, trained ONLY on bars from the
launch day onward (retrained nightly after the recorder stores the day's bars). Fixed warm-up:
no trades for the first 10 trading days; it also needs at least 5,000 training rows. So A and B
differ only in training data.

**C, random trader:** each decision bar from 09:45 to 14:25, enter with probability 0.02 a random
stock on a random side (random generator seeded from the date and bar number), 6-bar hold, same
limits. Fixed now; not matched to A's trade rate afterwards.

**D, cash:** holds nothing, earns the same 6.5%/yr. "Beats cash" means "beats D".

**Checkpoints: trading day 60 and 120 after launch.** An account WORKS only if ALL hold, on
ledger prices: (1) net return beats D; (2) net return beats C; (3) Sharpe of daily excess over D
> 0.5; (4) positive excess in both halves of the period; (5) at least 30 round trips. Latency
rule: the account must also beat D when each fill is re-priced at the quote taken at the decision
(live polling adds about a minute of delay that the ledger convention ignores). B's learning
check (report only): mean daily excess over D in days 61-120 vs days 11-60.

**Live feed rules:** bars come from 1-minute history through `resample_5m`, the still-forming bar
is dropped, the bars actually used are saved so any day can be replayed. Features use at least 25
prior trading days (tested equal to full-history features). With no valid Fyers token the runner
alerts and retries until about 09:30, then marks the day skipped and the accounts stay flat. If
positions from an earlier day are ever found, they are closed at that day's last close with an
alert. Request budget: the sandbox stays under about 30 Fyers requests/minute and off the
recorder's 5-minute option-chain slots. Untested until the first live login: the live feed, the
token probe and the request budget.

**2026-11-15, sandbox BUILT (not yet run live).** Code: `firm/features/intraday_model.py` (features,
labels, training), `firm/strategies/intraday_sandbox.py` (A/B model trader, C random, D cash),
`firm/live/sandbox.py` (runner, state saved every bar, overnight guard, model store),
`firm/live/sandbox_feed.py` (replay source), `firm/live/sandbox_live.py` (Fyers poller, token probe),
`firm/live/sandbox_eval.py` (day-60/120 rules incl. latency), `scripts/sandbox.py` (replay, live service,
`--shakedown`, startup entry). Verified: a replayed multi-day run equals a backtest; restart mid-day
resumes without double trading; stale overnight positions are closed with an alert; live bars equal
training-style bars and the still-forming bar is never used (fake-Fyers test); the short 25-day feature
window equals full-history features. Replay of two real days (12-13 Nov 2026, models trained only on
earlier data): A and B made 0 entries (threshold 0.25% never reached; valid, NOT loosened), C lost Rs 232
on 3 random trades, D earned the idle-cash interest. NOT verified until the first Fyers login: the live
feed against the real API, the token probe against a real expired token, and the request budget. Fyers
limits seen are per second and per minute only (no daily cap in the messages).

**2026-11-15, sandbox PRE-LAUNCH AMENDMENTS (definitions and plumbing only; no strategy setting changed).**
1. Checkpoint counting: "trading day 60 / 120" means the 60th / 120th RECORDED session (a day on which all
   four accounts have equity rows). A skipped no-login day does not count; all accounts are flat that day.
2. Request budget: the sandbox paces itself at about 40 Fyers requests/minute (the code default); with the
   recorder's roughly 6 option-chain calls per 5 minutes that stays under the 50/minute plan limit. This
   replaces the earlier "about 30" wording.
3. Latency rule, tightened: a decision-time quote is logged only if the decision happens within 5 minutes of
   the bar's close (a late start catching up on old bars logs none), and the latency check counts only if at
   least 90% of an account's fills have a logged quote; otherwise the account cannot "work" at that checkpoint.
4. Bug fixed before any live day: a Monday history gap (weekend) made the history patch ask Fyers for weekend
   bars and crash. Now only a missing previous TRADING day is fetched, and an empty Fyers response is handled.
5. MEASURED, informational: model A's predictions are tiny. Prediction std 0.016% against a label std of 0.31%
   (shrunk about 20x), 99.9th percentile of |prediction| about 0.12%. At the pre-registered 0.25% entry
   threshold: about 0.1 qualifying stock-bars/day in-sample and 0 out-of-sample (Oct 1 to Nov 13, model
   trained before Oct 1). So A and B are expected to make essentially NO trades unless the threshold is
   amended; at day 60 they would then read "not working: too few trades" by construction. The model is also
   saying that no trade is expected to beat the ~0.17% round-trip cost. DECISION PENDING with the owner before
   launch on 23 Nov: keep 0.25% (A and B stay flat, the test degenerates to C vs D), or a dated amendment based
   only on prediction scale (not on returns). Until amended, 0.25% stands.

**2026-11-15, sandbox AMENDMENT 2 (owner's go, before launch; supersedes the entry threshold only).**
The fixed 0.25% entry bar is replaced by a per-model bar: a model enters when |predicted 6-bar excess
return| >= the 99.9th percentile of |prediction| over THAT MODEL'S OWN TRAINING SAMPLE, recomputed at every
retrain (A monthly, B nightly) and stored with the model. Set from prediction scale only, never from returns.
Everything else (features, label, params, max 2 entries/bar, 4 positions, 6-bar hold, C, D, checkpoints,
latency rule) is unchanged. Disclosure: this amendment was prompted by seeing, on two replayed days and on
a prediction-scale check, that the 0.25% bar would have kept A and B flat; no performance number was used.
The launch (23 Nov) has not happened, so no forward data has been seen. Effect: the bar is about 0.1% in
terms of predicted return, BELOW the ~0.17% round-trip cost, so A and B will knowingly trade when their own
forecast says the trade loses after costs. That is deliberate: it tests whether the model's RANKING of
stocks beats random (C) after costs. Expect roughly 1-3 entries a day per account; a net cost drag of about
15%/yr on capital is the price of the test, and "A and B lose after costs" is a likely, valid outcome.
The checkpoint rules still decide whether anything works.

**Intraday Options: PARKED (owner's go, 2026-11-15).** No intraday option price history exists; the recorder
keeps saving real chains from the first market day with a Fyers login, revisit around May 2027.

## Overnight Options: phases 1-2 PRE-REGISTERED 2026-11-15, before any look at the option data

Data: NSE daily F&O bhavcopy, Nifty index options that traded, 1 Jan 2019 to 13 Nov 2026 (weekly-expiry era),
downloaded to data/fo/raw. Research window 1 Jan 2019 to 13 May 2026; the final 6 months (14 May 2026 on) are
LOCKED for a one-shot final exam that happens only if a baseline passes research, under its own
pre-registration. Bank Nifty is excluded (an at-the-money lot exceeds the 10% position cap at Rs 1L).

**Trade mechanics (all three baselines):** decide at day t's close; buy at that contract's day-t CLOSE
(bhavcopy close = volume-weighted price of the last 30 minutes) plus 0.5% slippage; sell at the next trading
day's OPEN minus 0.5% slippage. Contract: strike nearest to the Nifty close (50-point grid), nearest expiry
with at least 2 calendar days left, call or put per the rule. Skip the night if that contract has no day-t
trade, a zero close, or no next-day open. One position at a time, sized at 10% of current equity (fractional
lots: lot-size history is not modelled, real lots at Rs 1L are lumpier); the rest earns 6.5%/yr. Costs: the
firm's option cost model on each leg's premium turnover, today's rates in every year. Spot = Nifty daily close.

**Baselines (3 trials, track "options_overnight"):**
- O1 every night buy the at-the-money CALL (is the overnight drift tradable through options?)
- O2 call if the Nifty close is above its 50-day average, otherwise a PUT
- O3 buy a CALL after a day when the Nifty fell 1% or more, otherwise no trade

**Pass rule (research window):** ALL of: CAGR above cash (6.5%); Sharpe above 0.5; max drawdown better than
-25%; at least 60% of calendar years beating cash; at least 100 trades. Deflated Sharpe is reported, not a gate.
Also reported: the same run with zero costs and slippage, per-trade win rate and average result.
A baseline that passes becomes a candidate for phase 3 (a LightGBM skip filter and the locked exam, each under
its own pre-registration). If none passes, the team is parked with the finding recorded. Writing, running and
reading this study adds exactly these 3 trials; no parameter is tuned after a result.

**2026-11-15, sandbox AMENDMENT 3 (plumbing bug fix, before launch; applies to A, B and C equally).**
A replay on real data showed model A "entering" a short in POWERINDIA (about Rs 31,600 a share, above the
Rs 24,000 position slot): the order rounded to zero shares and vanished, so the account's books held a phantom
position and counted an entry that never traded. Fix: a stock costing more than one slot (Rs 24,000) per share
is never picked by A, B or C. Nothing else changes. Replay lesson: an account's own entry count must match real
fills; the checkpoint rule's "30 round trips" is counted from real fills only.

**2026-11-15, OVERNIGHT OPTIONS RESULT: all 3 baselines FAILED the pre-registered rule; team PARKED.**
Real NSE option prices, 1 Jan 2019 to 13 May 2026 (1,816 trading days, 2.87M contract-days), Rs 1L, 10% per
night. 3 trials logged (track options_overnight); locked window (14 May 2026 on) untouched; no final exam.

| | O1 always call | O2 trend call/put | O3 call after -1% day |
|---|---|---|---|
| Trades | 1,810 | 1,810 | 200 |
| Yearly before costs / Sharpe | 76.7% / 1.33 | 33.3% / 0.68 | 3.8% / -0.08 |
| Yearly after costs / Sharpe | 26.0% / 0.59 | -11.7% / -0.15 | -0.6% / -0.35 |
| Max drawdown | -62.3% | -93.9% | -43.9% |
| Years beating cash | 50% | 38% | 25% |
| Checks (CAGR, Sharpe, DD, years, trades) | Y Y n n Y | n n n n Y | n n n n Y |

O1 is the closest and needs care. Its yearly results: 2019 +21%, 2020 +111%, 2021 +403%, 2022 -34%, 2023 +31%,
2024 +5%, 2025 -38%, 2026 to 13 May -27% (cash about +6.5%). The 26% headline comes from the 2020-2021
bull/high-volatility years; the last three years lost money, so it looks like leveraged exposure to one regime,
not a steady edge. It also fails drawdown (-62%) and years beating cash (50%). Average premium move per night was
+1.42% before fees on an average Nifty move of +0.05%; fees (about 0.7% of premium per round trip) halve that, and
the result is sensitive to the assumed 0.5% slippage per side (real opening spreads may be wider; ATM lot sizes
at Rs 1L are lumpier than the fractional sizing used). Deflated Sharpe 0.95 for O1 counts only 3 trials. Decision
per the pre-registration: Overnight Options parked. Any follow-up (for example a smaller position size to cut the
drawdown) would be a NEW trial under its own pre-registration, followed by a one-shot final exam on the locked
window; nothing is run without the owner's choice.

## Control panel (dashboard page "Control"), built 2026-11-15

Everything operable from the browser: master halt, start/stop/restart of the recorder, the sandbox service and
Ollama, an on/off switch per data job and agent, pause per book, skip-a-sandbox-day, Run-now buttons, big jobs
(backfills, sandbox replay, the uncounted shake-down) and the Fyers login (open link, paste address, token saved
and never shown). A top strip says "all normal" or lists what needs attention with a one-click fix.
Design rules: (1) switches live in `data/control.db` (switches, command queue, audit log, pause windows), always
the live board whatever data mode the sidebar shows; every switch defaults to today's behaviour. (2) Pause and
halt only BLOCK NEW ENTRIES: books still mark to market, record equity and count their days, so nothing is
replayed on resume and the sandbox day-60/120 count is unaffected. Rejections caused by the owner say "paused by
the owner" and are NEVER counted as probation risk breaches; pause windows are shown on the trackers. (3) LOCKED
and shown read-only with the reason: paper mode, entry thresholds and model settings, risk limits, probation and
sandbox rules, final-exam windows. They change only by a dated amendment here. There is no real-order switch.
(4) The AI Trader guard can only be shadow or off. (5) Stopping a service is cooperative first (the stop flag),
forced by PID only as a fallback; stopping the sandbox mid-session needs typing STOP. (6) The dashboard is bound
to 127.0.0.1 only (it was reachable from the Wi-Fi before). (7) Long jobs run as their own detached processes so
they never freeze the recorder. Tests cover: disabled jobs do not run, commands run exactly once, halt/pause
semantics for swing and sandbox, owner pauses excluded from probation, login state check and secrecy, forced stop,
and every page section in every data mode.

## LLM layer rules (2026-11-14)

- LLM output is research only. It never places or sizes orders.
- LLM-derived signals are judged forward only (a local model may know past outcomes):
  Researcher notes scored on the next 5 trading days from the next open vs equal-weight.
- Manager notes: every number must appear in the code-built facts, else a template is used.
- AI Trader guard runs in shadow mode during probation (logged, scored, not applied).
- Weekly strategist = Claude in Claude Code (`/weekly-review`), discusses before running anything.

## Head v0 (shadow), Alerts, Backups: built 2026-11-15

**Head v0 (advice only).** Reads every book each evening and writes one firm report plus an allocation
proposal (`head_reports` table, shown on Home). Pure code, no LLM, so the text cannot contain invented numbers.
It moves no money and imports no broker, risk or runner code (a test enforces it). Rules: a team gets capital
only after it passes its own pre-registered test (swing: the probation promote rule above, incl. the manual
trades-match check; intraday stocks: a sandbox account passes a day-60 or day-120 checkpoint). With none
passed the proposal is HOLD. If some pass: equal split, at most 50% each, rest cash. Parked teams get 0.
Making the Head act on its proposal would need a new pre-registered decision after real results exist.

**Alerts.** Windows notification (plus optional ntfy phone push) on trading days at 08:45, 09:00, 09:10 if the
Fyers login is missing, and once if the practice-trading program is not running at 09:05 to 09:30.

**Fyers daily login.** Fyers' refresh-token endpoint is disabled ("to comply with SEBI regulations",
checked with a real call on 2026-11-15), so a manual login is needed every trading day. Auto-renew code is
kept dormant; it does nothing without a stored PIN.

**Backups.** Daily after 18:00: databases via SQLite's backup call, plain files mirrored, nothing deleted,
`.env` never copied. Default folder D:\TradingFirmBackups (change it in Settings). A separate drive letter on
the same physical disk does not protect against disk failure: point it at an external drive or OneDrive.

**NSE holiday list.** Fetched daily; 2027's list is picked up automatically once NSE publishes it.

## Open issues

- Survivorship bias in the stock universe (historical index membership not free).
- Corporate actions handled by a heuristic (moves over 40% treated as unadjusted breaks
  unless listed in `KNOWN_REAL_MOVES`). Replace with NSE corporate-actions feed later.

- Disk: ~16GB free on C. Model + recorded data will outgrow it.
- Laptop must stay on and awake during market hours.
