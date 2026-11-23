# crytrd — Crypto Swing Signal System (Project Plan)

Handoff document. Any new Claude Code session (e.g. in VS Code): **read this file first**, then continue from "Next step".

## Goal

A system that analyses the crypto market in real time, **selects coins itself** from a liquid universe, and predicts the
probability of a profitable swing move. It sends **Telegram alerts** and shows a **web dashboard**.
**The user trades manually on CoinDCX** — the bot never places real orders.

Framing: the model does not "predict the price". It answers:
> "What is the probability this coin rises ≥ X% within the next 1–3 days, net of fees?"
and ranks coins by that probability, flagging only high-confidence setups. "No signal" is a valid output.

## User context & decisions

| Topic | Decision |
|---|---|
| Location / exchange | India, trades on **CoinDCX** (INR / USDT pairs), **manual execution** |
| Style | **Swing trading, 1–3 day holds** (chosen over scalping: better signal/noise, fees matter less) |
| Universe | **Top 50 coins by volume**, excluding stablecoins, leveraged tokens, very new listings |
| Market | **Spot only, no leverage** |
| Output | **Web dashboard** (FreqUI) + **Telegram alerts** (signals + notable coin movements) |
| Hardware | Windows 11 laptop, **RTX 2050 (4 GB VRAM)**, Docker Desktop + WSL2 available |
| Budget | Free / open-source tools only (optional VPS later ~₹400–800/mo for 24/7 uptime) |
| Philosophy | Reuse existing tools, don't build from scratch |

## Architecture

```
Binance public data ─┐
News + local LLM ────┼──► FreqAI model (LightGBM + Chronos features) ──► ranked coin signals
Funding / OI ────────┘                                                    │
                                    ┌─────────────────────────────────────┼──────────────┐
                                  FreqUI                              Telegram         Paper log
                               (dashboard)                            alerts        (proof of edge)
                                                                         │
                                                              User trades manually on CoinDCX
```

### Stack (all free)

| Layer | Tool |
|---|---|
| Bot framework | **Freqtrade** (Docker image), always in **dry-run** mode |
| ML | **FreqAI** with LightGBM / XGBoost, auto-retraining every few days |
| Coin selection | Freqtrade `VolumePairList` (50 pairs) + filters (spread, age, price, volatility) |
| Market data | **Binance public data** (signal source — deepest liquidity/history). CoinDCX prices track global + small INR premium |
| Dashboard | **FreqUI** on localhost |
| Alerts | Freqtrade built-in **Telegram bot** (switched from Discord 2026-11-23) (+ custom movement alerts later) |
| Time-series models (Phase 2) | **Chronos-Bolt** small/base or **TimesFM** as extra features |
| Derivatives data (Phase 2) | Binance futures funding rates + open interest (free) |
| News (Phase 3) | **CryptoPanic API** free tier (verify current terms) + RSS |
| Sentiment (Phase 3) | Local LLM via **Ollama** (or **OpenJarvis**): Qwen2.5-3B / Llama 3.2 3B / Phi-3.5-mini at Q4; or CryptoBERT/FinBERT |
| DeFi / unlocks (Phase 3) | **DefiLlama** (free) |

### GPU limits (RTX 2050, 4 GB)
- LightGBM/XGBoost: CPU, fine.
- Chronos-Bolt small/base: fits.
- LLM: ~3–4B params quantized only; load on demand, don't run alongside everything else. 7B partially offloads → slow.
- Swing timeframe means speed is not critical (signals update on 1h/4h candle close).

### CoinDCX note
CoinDCX is likely **not supported by ccxt/Freqtrade** (verify). Plan: signals from Binance data, manual trades on CoinDCX.
Optional Phase 4: small adapter using CoinDCX public REST API for INR prices / premium.

## India-specific constraints (affect strategy design)
- **30% flat tax** on crypto gains, **no loss set-off** → favour fewer, higher-confidence trades; strict confidence threshold.
- **1% TDS** on sales above threshold → ties up capital between trades.
- Spot only, no leverage until paper results prove the edge.

## Validation rules (non-negotiable)
- **Walk-forward** evaluation only; never random shuffles.
- Compare against the **base rate** (how often coins rise ≥ X% anyway).
- Include **fees, spread, slippage** in every backtest.
- **Live paper log** of every signal for 4–8 weeks before trusting real money.
- User also logs **their own CoinDCX fills** to compare vs. bot's paper trades (INR premium, timing, slippage).
- Retrain regularly; expect edges to decay.

## Roadmap

### Phase 1 — Base system (NEXT)
- Freqtrade + FreqUI via Docker (`docker compose`), dry-run, Binance data
- Pairlist: top 50 by volume + filters
- Timeframes: 1h and 4h
- FreqAI target: P(max/close return ≥ ~5% within 1–3 days) — tune threshold
- Features: multi-window returns, volatility (ATR), RSI, MACD, Bollinger position, volume z-score, BTC-relative strength
- LightGBM classifier, periodic retraining
- Backtest last 1–2 years with fees; report precision of top-confidence picks vs base rate
- Telegram alerts: coin, entry zone, stop-loss (ATR-based), target, confidence, expected hold time
- FreqUI dashboard on localhost

### Phase 2 — Market structure features
- Funding rate + open interest features
- Chronos-Bolt forecasts as features
- Custom Telegram "big move" alerts (volume spikes, ±X% moves on watchlist) independent of model signals

### Phase 3 — News & LLM
- CryptoPanic + RSS ingestion
- Ollama/OpenJarvis sentiment scoring per coin → feature for FreqAI
- LLM-generated "why this coin" explanation in each alert
- Event flags: hacks, delistings, listings, token unlocks (DefiLlama)

### Phase 4 — Only if paper results hold 4–8 weeks
- CoinDCX price adapter (INR premium)
- Possibly semi-automated execution (user decision, not default)

## What the user provides
- Docker Desktop running
- Telegram bot token (@BotFather) + chat id — store in a git-ignored config/env file, never commit
- CryptoPanic account/API key (Phase 3) — user signs up
- Approval for large downloads (Docker images, model weights)
- **No exchange API keys are needed** (signals-only system)

## Boundaries
- The system is the user's own rule-based/ML tool; Claude builds and maintains it but does **not** give personal buy/sell advice or send trading calls itself.
- No real order execution. No guarantee of profit.

## Next step
**Phase 1 scaffolded (2026-11-23)**, not yet run: `docker-compose.yml`, `user_data/config.json` (+ `config.backtest.json`
overlay), strategy `SwingFreqAI.py`, `user_data/scripts/evaluate_signals.py`, `README.md` with commands.
Blocked on: **Docker Desktop is not installed** on the laptop.
Then: `.env` from `.env.example` → `docker compose pull` → download data → first backtest → evaluate precision vs
base rate → tune `entry_threshold` / target. Possible label upgrade: "target hit before ATR stop" instead of plain max-return.
