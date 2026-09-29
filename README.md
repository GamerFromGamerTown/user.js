# BTC 5m / 15m Up-Down classifier for Polymarket

Predicts whether BTC/USD closes a Polymarket 5-minute or 15-minute window at or above its
opening price ("Up" resolves on `>=`), and runs a paper-only live estimator that compares
the model's fair value with the Polymarket order book.

## Data

The session's network policy blocked every exchange, Polymarket and Chainlink endpoint, so
the only bulk source reachable was
[ff137/bitstamp-btcusd-minute-data](https://github.com/ff137/bitstamp-btcusd-minute-data):
Bitstamp BTC/USD 1-minute candles, 2012 → 2026-09-29 (6.7 M minutes used, from 2014).
ETH, LTC and XMR history could not be downloaded, so the "massive pretraining" corpus is
the full BTC history rather than multi-asset data.

Clone it and point `BTC_MINUTE_REPO` at it (default `/home/user/ff137/bitstamp-btcusd-minute-data`).

## Splits (strictly chronological, 1-day embargo at each edge)

| split    | period                  | role                                   |
|----------|-------------------------|----------------------------------------|
| pretrain | 2014-03 → 2022-12       | NN pretraining                         |
| finetune | 2023-01 → 2025-09       | NN fine-tuning, GBM training           |
| val      | 2025-10 → 2026-03       | early stopping, ensemble weight        |
| test     | 2026-04 → 2026-09-28    | untouched backtest                     |

Evaluation uses only Polymarket-aligned windows (start at :00/:05/… for 5m, :00/:15/… for 15m).

## Models

* `train_gbm.py` – LightGBM on 44 causal features (multi-lag vol-normalised returns, EMA
  deviations, realised-vol regime, range position, volume and signed-volume, time of day /
  week, position in the 15-minute block).
* `train_nn.py` – tabular MLP + dilated 1-D CNN over the last 64 minutes, multi-task
  (Up@5m, Up@15m, auxiliary return regression). Pretrained on 2014–2022, fine-tuned on
  2023–2025Q3.
* `ensemble.py` – blends the two with a weight chosen on validation log-loss; writes
  `models/ensemble.json` and `logs/backtest.json`.
* `walkforward.py` – retrains the GBM on earlier windows and scores the following 6 months.
* `refit_live.py` – refits the GBMs on 2023-01 → latest data into `models/live/`; the live
  runner uses these (the NN and ensemble weights are unchanged from the backtest).

## Results (test set, 2026-04-01 → 2026-09-28)

| horizon | test windows | accuracy (±1 SE) | top 50 % | top 20 % | top 10 % | Up base rate |
|---|---|---|---|---|---|---|
| 5m | 51,870 | **52.46 %** (±0.22) | 54.0 % | 56.1 % | 57.1 % | 50.12 % |
| 15m | 17,290 | **52.75 %** (±0.38) | 54.7 % | 56.2 % | 55.3 % | 49.70 % |

Component models on the same test set: GBM 52.30 % / 52.88 %, NN 52.04 % / 52.48 % (5m / 15m).
Walk-forward GBM refits on four earlier 6-month periods (2023-10 → 2025-09): 5m 51.4–52.3 %,
15m 52.2–53.7 % (`walkforward.py`, `logs/walkforward.log`).

"Top 10 %" is accuracy on the tenth of windows where the model is most confident, i.e. the
windows a bettor would actually trade.

## Caveats

* Trained and tested on Bitstamp last-trade prices. Polymarket settles on the Chainlink
  BTC/USD Data Stream (a cross-exchange aggregate); part of any edge that comes from
  Bitstamp microstructure will not transfer. Lagging all features by 10 minutes still left
  51.4 % test accuracy on 5m, so most of the signal is not last-minute noise.
* The 50 % baseline ignores Polymarket prices: the market often prices Up away from 0.50,
  and taker fees plus the spread must be beaten. No historical Polymarket books were
  reachable, so edge versus the actual market is only measured by the live runner.
* Accuracy varies month to month (≈49.5–55 % on 5m); ±0.22 % is one standard error on the
  full 5m test set.

## Live estimator

`live.py` predicts every Polymarket BTC 5m / 15m window at its open, then every 15 s
compares its fair value (model prior blended with the move so far) with the Polymarket
best ask, net of the crypto taker fee (`shares × p × 0.25 × (p(1−p))²`). It paper-buys one
share when the edge after fees is at least `--min-edge` (default 0.03). No orders are placed.

### Running on GitHub Actions (no computer needed)

`.github/workflows/live.yml` chains 100 stints of about 6 hours each (≈25 days) on GitHub's
runners. Each stint restores the previous stint's state from the committed logs.

* **Start:** any push to this branch that changes `live.py`, `btcpred/`, `models/` or
  `.github/` starts a new run and cancels the old one. After a run ends, open it in the
  Actions tab and choose *Re-run all jobs*.
* **Stop:** Actions tab → the running *live* workflow → *Cancel workflow run*.
* **Results:** committed back to this branch every 15 minutes:
  * `logs/live_summary.json`: running accuracy (exchange-based and Polymarket-resolved),
    paper trades, wins and PnL after fees
  * `logs/live_predictions.csv`: one row per window (prediction and both outcomes)
  * `logs/quotes/YYYY-MM-DD.csv`: fair value vs. best asks and edge, every 15 s
  * `logs/live_stdout.txt`: runner log

Actions on a fork may need enabling once (Actions tab → enable workflows). GitHub's terms
limit Actions to work related to the repository's software; a long-running monitor is a
gray area and GitHub may disable it.

### Running locally

```
pip install -r requirements.txt
python live.py --min-edge 0.03 --poll 15
```

Needs outbound access to `www.bitstamp.net` (or `api.exchange.coinbase.com` /
`api.binance.com`), `gamma-api.polymarket.com` and `clob.polymarket.com`.

## Rebuild from scratch

```
python -m btcpred.data          # 1-minute grid cache
python train_gbm.py             # GBM
python train_nn.py 3 5          # NN: pretrain epochs, fine-tune epochs
python ensemble.py              # blend + backtest
python walkforward.py           # optional robustness check
python refit_live.py            # refit GBMs on all recent data for live use
```
