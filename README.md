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

* **Pre-window ensemble** (`train_final.py`): five LightGBM variants on 75 causal features
  (44 original + 31 added: last ten 1-minute moves, move since the day / hour / quarter-hour
  opened, multi-day trend, distance to round prices, up-minute share, skew, autocorrelation,
  volume detail) plus the neural net (`train_nn.py`: tabular MLP + dilated 1-D CNN over the last
  64 minutes, pretrained on 2014–2022 BTC, fine-tuned on 2023–2025Q3). Member logits are averaged,
  blended with the NN (weight chosen on validation) and Platt-calibrated on validation.
* **Partial-window model** (`train_intra.py`, `btcpred/intra.py`): P(Up) for a window already in
  progress, from the regular features plus the move since the open measured against the
  volatility left (`disp_z`), elapsed and remaining time.
* `experiments.py` holds the variants tried (features, regression / weighted targets, tree
  sizes, training spans, recency weighting, extra-trees, aligned-only training); all landed
  within noise of each other on validation (5m 52.2–52.7 %, 15m 53.4–53.9 %).

## Results (test set, 2026-04-01 → 2026-09-28, never used for choices)

### At the window open

| window | test windows | accuracy (±1 SE) | log-loss | calibration error (ECE) |
|---|---|---|---|---|
| 5m | 51,870 | **52.41 %** (±0.22) | 0.6909 | 0.46 % |
| 15m | 17,290 | **52.76 %** (±0.38) | 0.6909 | 1.39 % |

Overall test accuracy is unchanged from the first version (52.46 % / 52.75 %): the new features
and the blend improved validation (52.9 % / 53.7 %) but not the test period. What improved is
the probabilities: they are calibrated, and the confidence tiers separate well.

**Confidence tiers** (thresholds set on validation):

| tier | share of windows | 5m accuracy | 15m accuracy |
|---|---|---|---|
| high | 10 % | 57.5 % | 57.2 % |
| medium | 20 % | 54.5 % | 54.6 % |
| low | 30 % | 52.3 % | 53.1 % |
| none | 40 % | 50.2 % | 50.4 % |

**Accuracy vs. how selective you are** — the realistic route to 54 %+:

| trade only the most confident … | 100 % | 75 % | 50 % | 30 % | 20 % | 10 % | 5 % |
|---|---|---|---|---|---|---|---|
| 5m accuracy | 52.4 % | 53.2 % | 54.3 % | 55.5 % | 56.1 % | 57.2 % | 57.9 % |
| 15m accuracy | 52.8 % | 53.6 % | 54.6 % | 55.5 % | 56.7 % | 57.0 % | 59.5 % |

### Part-way through a window

| elapsed | 5m acc | 15m acc | 15m acc, analytic baseline |
|---|---|---|---|
| 1 min | 65.6 % | 59.7 % | 59.5 % |
| 2 min | 72.6 % | 63.2 % | 63.5 % |
| 3 min | 79.2 % | 66.6 % | 67.1 % |
| 4 min | 86.3 % | 68.8 % | 69.0 % |
| 7 min | – | 76.0 % | 76.2 % |
| 10 min | – | 82.7 % | 82.8 % |
| 14 min | – | 93.4 % | 93.5 % |

The partial-window model ranks outcomes about as well as the analytic formula
Φ(move / (σ·√time left)) but its probabilities are better (lower log-loss at every minute,
calibration error 0.3–2 %), which is what matters when comparing with Polymarket prices.

### What the predictor returns

`Predictor.predict()` → per window: `p_up` (calibrated), `side`, `confidence` = P(predicted side),
`tier` (high / medium / low / none) with that tier's backtest accuracy, and `spread` (standard
deviation of the six members' probabilities: high spread = the models disagree).
`Predictor.partial(move, elapsed_min, H)` → P(Up) for a window in progress.

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

`live.py` predicts every Polymarket BTC 5m / 15m window at its open (calibrated P(Up),
confidence tier, model spread). Once a minute it refreshes candles and features; every 15 s it
turns the move so far into a fair value with the partial-window model, compares it with the
Polymarket best ask net of the crypto taker fee (`shares × p × 0.25 × (p(1−p))²`), and logs the
edge and a Kelly stake fraction. It paper-buys one share when the edge after fees is at least
`--min-edge` (default 0.03). `logs/live_summary.json` tracks accuracy overall, per tier, against
Polymarket's own resolution, the Brier score, a rolling 500-window accuracy and a drift warning.
No orders are placed.

### Making it work better live

1. **Settle on Chainlink, not Bitstamp.** Polymarket resolves on the Chainlink BTC/USD Data
   Stream. Reading it from Polymarket's live-data websocket (`ws-live-data.polymarket.com`,
   topic `crypto_prices_chainlink`) for the opening price and the move so far removes basis noise
   that decides close calls late in a window.
2. **Lower latency.** Replace 15 s REST polling with exchange websockets (mid price, not last
   trade); late in a window a few seconds of lag is the whole edge.
3. **Record order-book and flow data** (top-of-book imbalance, trade flow, perp basis/funding)
   and retrain after a few weeks; these are the strongest short-horizon signals and are absent
   from the historical data used here.
4. **Only act on high/medium tiers.** The bottom 40 % of windows is a coin flip.
5. **Use limit (maker) orders.** Taker fees reach 1.56 % at 50 ¢, larger than much of the edge;
   makers pay no fee and earn rebates.
6. **Size with fractional Kelly** (a quarter of `kelly_up` / `kelly_down` in the quotes log).
7. **Retrain weekly** (`train_final.py`, `train_intra.py` refit to the latest data); accuracy
   drifts month to month.
8. **Stop when `drift_warning` fires** in `live_summary.json` (rolling accuracy < 49 % over
   300+ windows).
9. **Judge it against the market, not 50 %.** After a few weeks, `logs/quotes/` gives the real
   benchmark: did buying when fair − ask − fee > 0 make money?
10. **Add other coins** (ETH, SOL, …) for pretraining and as cross-asset features once a data
    host such as `data.binance.vision` is reachable.

### Running on GitHub Actions (no computer needed)

`.github/workflows/live.yml` runs 100 stints of about 6 hours each (≈25 days) one after
another on GitHub's runners (a matrix with `max-parallel: 1`). Each stint restores the
previous stint's state from the committed logs.

* **Start:** a push to this branch that changes `live.py`, `btcpred/`, `models/` or the
  workflow starts a new run and cancels the old one. After a run ends, open it in the
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
python train_nn.py 3 5 0 2      # NN: pretrain epochs, fine-tune epochs, seed, pretrain stride
python train_final.py           # GBM members + blend + calibration + tiers + live refits
python train_intra.py           # partial-window models (+ live refits)
python experiments.py           # optional: the variants compared on validation
```
