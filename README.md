# crytrd — crypto swing signals (Phase 1)

Freqtrade + FreqAI (LightGBM) in Docker, **dry-run only**, Binance public data.
Paper entries are posted to Discord as signals; you execute manually on CoinDCX if you choose.
See [PLAN.md](PLAN.md) for the full design.

## Layout

| Path | Purpose |
|---|---|
| `docker-compose.yml` | Freqtrade (`stable_freqai` image), FreqUI on `127.0.0.1:8080` |
| `.env` (from `.env.example`) | Discord webhook + FreqUI login. Git-ignored |
| `user_data/config.json` | Dry-run, top-50 VolumePairList + filters, FreqAI, Discord, API server |
| `user_data/config.backtest.json` | Backtest overlay: static list of 35 liquid pairs |
| `user_data/strategies/SwingFreqAI.py` | Features, target, entry threshold, ATR stop, 72h time exit |
| `user_data/scripts/evaluate_signals.py` | Precision of high-confidence picks vs base rate |

## Model in one paragraph

On every 1h candle, for every pair: *P(high touches ≥ 5.6% above close within 72h)*. 5.6% = 5% net target +
0.6% round-trip cost. Features: returns, RSI, NATR, Bollinger position and width, volume z-score, distance from
recent high/low (periods 14/42, 1h + 4h, with 2 shifted candles). Each feature is also computed for BTC and ETH,
plus MACD, day/hour, and strength relative to BTC over 1/3/7 days. FreqAI trains on 180 days and retrains every 48h
live; in backtests it walks forward in 14-day blocks. Entry when P ≥ `entry_threshold` (default 0.65).
Exits: +5% net (ROI), stop at 2×ATR (clamped 2–10%), or 72h time exit.

## Setup

1. Install **Docker Desktop** (WSL2 backend) and start it.
2. `copy .env.example .env`, then fill in the Discord webhook URL and FreqUI password/secrets.
3. Pull the image (~2 GB): `docker compose pull`

## Backtest (do this before running live)

```powershell
# 1. History: 1h + 4h. FreqAI needs 180 days *before* the test start, so download from early 2024.
docker compose run --rm freqtrade download-data --config user_data/config.json --config user_data/config.backtest.json --timeframes 1h 4h --timerange 20240301-

# 2. Walk-forward backtest over ~2 years (slow: one model per pair per 14 days; expect an hour or more).
docker compose run --rm freqtrade backtesting --config user_data/config.json --config user_data/config.backtest.json --strategy SwingFreqAI --freqaimodel LightGBMClassifier --timerange 20241101-20261115 --export trades

# 3. Precision vs base rate, per threshold, top-k per day, per month
docker compose run --rm --entrypoint python freqtrade user_data/scripts/evaluate_signals.py
```

What to look for: precision at the chosen threshold clearly above the base rate (lift), **stable across months**,
and a positive backtest after the 0.3% fee. If lift ≈ 1, the model has no edge. That is a valid result.

Notes:
- FreqAI caches backtest models under `user_data/models/swing_lgbm_v1`. **Change `identifier`** in config.json after
  changing features/target, or old predictions get reused.
- `fee: 0.003` is a placeholder for CoinDCX fee + slippage. Set it to your actual CoinDCX tier.
- The static backtest list has survivorship bias (coins that are still big today).

## Run live (paper)

```powershell
docker compose up -d
docker compose logs -f
```

FreqUI: http://localhost:8080 (login from `.env`). The first models train on startup, which takes several minutes
for 50 pairs. Discord gets a message on each paper entry (confidence | entry zone | stop | target | hold) and exit.
With `max_open_trades: 5`, new signals are not posted while 5 paper trades are open.
