"""Live BTC Up/Down estimator for Polymarket 5m / 15m markets. Runs until killed.

At every 5-minute boundary it predicts P(Up) for the window that just opened (and the
15-minute window on quarter hours), then, every POLL seconds inside each window, blends the
prior with the move so far into a fair value and compares it with the Polymarket order
book. Everything is paper-only: it logs predictions, outcomes, running accuracy and the
PnL of hypothetical 1-share buys whenever edge = fair - ask exceeds --min-edge.

Price feed: Bitstamp (matches the training data), falling back to Coinbase, then Binance.
Polymarket resolves on the Chainlink BTC/USD stream; exchange prices are a proxy for it.

    python live.py [--min-edge 0.03] [--poll 15]
Outputs: logs/live_predictions.csv, logs/live_quotes.csv, logs/live_summary.json
"""
import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
import numpy as np
import requests
from btcpred.predictor import Predictor

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
HIST = 3000  # minutes of history the features need
S = requests.Session()
S.headers["User-Agent"] = "btcpred/1.0"


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), *a, flush=True)


# ---------------------------------------------------------------- price feeds
def _bitstamp(end):
    rows = {}
    t = end
    while len(rows) < HIST:
        r = S.get("https://www.bitstamp.net/api/v2/ohlc/btcusd/",
                  params={"step": 60, "limit": 1000, "end": t}, timeout=10)
        r.raise_for_status()
        data = r.json()["data"]["ohlc"]
        if not data:
            break
        for k in data:
            rows[int(k["timestamp"])] = [float(k[x]) for x in ("open", "high", "low", "close", "volume")]
        t = min(rows) - 60
    return rows


def _coinbase(end):
    rows = {}
    t = end
    while len(rows) < HIST:
        a = t - 300 * 60
        r = S.get("https://api.exchange.coinbase.com/products/BTC-USD/candles",
                  params={"granularity": 60, "start": a, "end": t}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        for ts_, lo, hi, op, cl, vo in data:
            rows[int(ts_)] = [op, hi, lo, cl, vo]
        t = a
    return rows


def _binance(end):
    rows = {}
    t = end * 1000
    while len(rows) < HIST:
        r = S.get("https://api.binance.com/api/v3/klines",
                  params={"symbol": "BTCUSDT", "interval": "1m", "limit": 1000, "endTime": t}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        for k in data:
            rows[int(k[0]) // 1000] = [float(x) for x in k[1:6]]
        t = min(rows) * 1000 - 1
    return rows


FEEDS = [("bitstamp", _bitstamp), ("coinbase", _coinbase), ("binance", _binance)]


def candles(now):
    """Closed 1-minute candles up to `now`, as gap-free arrays with ts = candle close time."""
    last_err = None
    for name, fn in FEEDS:
        try:
            rows = fn(int(now))
            opens = sorted(t for t in rows if t + 60 <= now)[-HIST:]
            if len(opens) < HIST - 60:
                raise RuntimeError(f"only {len(opens)} candles")
            t0, t1 = opens[0], opens[-1]
            n = (t1 - t0) // 60 + 1
            a = np.full((n, 5), np.nan)
            for t in opens:
                a[(t - t0) // 60] = rows[t]
            for j in range(n):  # forward-fill missing minutes with a flat, zero-volume candle
                if np.isnan(a[j, 0]):
                    a[j, :4] = a[j - 1, 3]
                    a[j, 4] = 0.0
            return name, {"ts": t0 + 60 + 60 * np.arange(n), "o": a[:, 0], "h": a[:, 1],
                          "l": a[:, 2], "c": a[:, 3], "v": a[:, 4]}
        except Exception as e:  # try the next feed
            last_err = f"{name}: {e}"
    raise RuntimeError(f"all price feeds failed ({last_err})")


def spot():
    for url, get in (("https://www.bitstamp.net/api/v2/ticker/btcusd/", lambda j: j["last"]),
                     ("https://api.exchange.coinbase.com/products/BTC-USD/ticker", lambda j: j["price"]),
                     ("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", lambda j: j["price"])):
        try:
            return float(get(S.get(url, timeout=5).json()))
        except Exception:
            continue
    return None


# ---------------------------------------------------------------- polymarket
def pm_market(H, start):
    """Up/Down token ids for the window starting at `start` (unix s), or None."""
    for slug in (f"btc-updown-{H}m-{start}",):
        try:
            r = S.get("https://gamma-api.polymarket.com/markets", params={"slug": slug}, timeout=10)
            js = r.json()
            if js:
                m = js[0]
                ids = json.loads(m["clobTokenIds"])
                outs = [o.lower() for o in json.loads(m["outcomes"])]
                return {"slug": slug, "up": ids[outs.index("up")], "down": ids[outs.index("down")]}
        except Exception:
            pass
    return None


def pm_ask(token):
    """Best ask (price, size) for a token, or (None, None)."""
    try:
        b = S.get("https://clob.polymarket.com/book", params={"token_id": token}, timeout=5).json()
        asks = [(float(x["price"]), float(x["size"])) for x in b.get("asks", [])]
        return min(asks) if asks else (None, None)
    except Exception:
        return None, None


# ---------------------------------------------------------------- fair value
def fair_up(prior, s0, st, sigma1m, secs_left):
    """P(close >= s0) given the current price st, blending the model prior as a drift tilt.

    The prior at window open implies a drift z0 = Phi^-1(prior) over the full window; the
    remaining drift scales with the fraction of time left.
    """
    if secs_left <= 0:
        return 1.0 if st >= s0 else 0.0
    tau = secs_left / 60.0
    z0 = _ppf(min(max(prior, 1e-4), 1 - 1e-4))
    sd = sigma1m * math.sqrt(tau)
    return _cdf((math.log(st / s0) / sd) + z0 * math.sqrt(tau / fair_up.H))


fair_up.H = 5


def _cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _ppf(p):
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _cdf(mid) < p else (lo, mid)
    return (lo + hi) / 2


# ---------------------------------------------------------------- main loop
class Book:
    def __init__(self):
        os.makedirs(LOG, exist_ok=True)
        self.pred_path = os.path.join(LOG, "live_predictions.csv")
        self.quote_path = os.path.join(LOG, "live_quotes.csv")
        self.sum_path = os.path.join(LOG, "live_summary.json")
        self.open = {}  # (H, start) -> dict
        self.stats = {str(H): {"n": 0, "correct": 0, "trades": 0, "pnl": 0.0} for H in (5, 15)}
        if os.path.exists(self.sum_path):
            self.stats.update(json.load(open(self.sum_path)).get("stats", {}))

    def _append(self, path, row):
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    def save(self):
        out = {"updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stats": self.stats}
        for H, s in self.stats.items():
            s["accuracy"] = s["correct"] / s["n"] if s["n"] else None
        json.dump(out, open(self.sum_path, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-edge", type=float, default=0.03, help="fair - ask needed to paper-buy")
    ap.add_argument("--poll", type=int, default=15, help="seconds between in-window quotes")
    ap.add_argument("--once", action="store_true", help="make one prediction and exit")
    args = ap.parse_args()
    model = Predictor()
    book = Book()
    last_boundary = None
    log("started; models loaded")
    while True:
        try:
            now = time.time()
            boundary = int(now // 300 * 300)
            if boundary != last_boundary and 2 <= now - boundary < 120:
                feed, d = candles(boundary)
                last_boundary = boundary  # only after a successful fetch, so failures retry
                s0 = float(d["c"][-1])
                p = model.predict(d)
                # resolve windows that have ended
                for key in [k for k in book.open if k[1] + 60 * k[0] <= boundary]:
                    w = book.open.pop(key)
                    end = d["c"][d["ts"] == key[1] + 60 * key[0]]
                    if not len(end):
                        continue
                    up = int(end[0] >= w["s0"])
                    st = book.stats[str(key[0])]
                    st["n"] += 1
                    st["correct"] += int((w["p"] >= 0.5) == up)
                    for side, px in w.get("fills", []):
                        st["trades"] += 1
                        st["pnl"] += (1.0 if (side == "up") == bool(up) else 0.0) - px
                    book._append(book.pred_path, {"start": key[1], "horizon": key[0], "feed": w["feed"],
                                                  "s0": w["s0"], "s_end": float(end[0]), "p_up": round(w["p"], 4),
                                                  "outcome_up": up, "fills": json.dumps(w.get("fills", []))})
                for H in (5, 15):
                    if boundary % (60 * H) == 0:
                        book.open[(H, boundary)] = {"p": p[H], "s0": s0, "feed": feed, "sigma": p["sigma1m"],
                                                    "mkt": pm_market(H, boundary), "fills": []}
                        log(f"{H:>2}m window {time.strftime('%H:%M', time.gmtime(boundary))} "
                            f"P(up)={p[H]:.4f} s0={s0:.2f} feed={feed} "
                            f"polymarket={'found' if book.open[(H, boundary)]['mkt'] else 'n/a'}")
                book.save()
                if args.once:
                    return
            # in-window quotes vs Polymarket
            st_px = spot()
            for (H, start), w in list(book.open.items()):
                if st_px is None or not w["mkt"]:
                    continue
                left = start + 60 * H - time.time()
                fair_up.H = H
                fu = fair_up(w["p"], w["s0"], st_px, w["sigma"], left)
                au, _ = pm_ask(w["mkt"]["up"])
                ad, _ = pm_ask(w["mkt"]["down"])
                row = {"t": int(time.time()), "horizon": H, "start": start, "spot": st_px, "fair_up": round(fu, 4),
                       "ask_up": au, "ask_down": ad,
                       "edge_up": None if au is None else round(fu - au, 4),
                       "edge_down": None if ad is None else round((1 - fu) - ad, 4)}
                book._append(book.quote_path, row)
                sides = {s for s, _ in w["fills"]}
                for side, edge, px in (("up", row["edge_up"], au), ("down", row["edge_down"], ad)):
                    if edge is not None and edge >= args.min_edge and side not in sides and left > 10:
                        w["fills"].append((side, px))
                        log(f"paper BUY {side} {H}m @ {px} fair={fu:.3f} edge={edge:.3f}")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log("error:", e)
            if "--debug" in sys.argv:
                traceback.print_exc()
            time.sleep(20)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
