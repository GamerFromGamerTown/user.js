"""Live BTC Up/Down estimator for Polymarket 5m / 15m markets. Runs until killed.

At every 5-minute boundary it predicts a calibrated P(Up) for the window that just opened
(and the 15-minute window on quarter hours), with a confidence tier and the models'
disagreement. Every POLL seconds inside each window, the partial-window model turns the
move so far and the time left into a fair value, which is compared with the Polymarket
order book net of taker fees, together with a Kelly stake fraction. Everything is paper-only:
it logs predictions, outcomes, running accuracy (overall, per tier, rolling with a drift
warning) and the PnL of hypothetical 1-share buys whenever fair - ask - fee >= --min-edge.

Price feed: Bitstamp (matches the training data), falling back to Coinbase, then Binance.
Polymarket resolves on the Chainlink BTC/USD stream; exchange prices are a proxy for it.

    python live.py [--min-edge 0.03] [--poll 15] [--duration MINUTES]
Outputs: logs/live_predictions.csv, logs/quotes/YYYY-MM-DD.csv, logs/live_summary.json
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

LOG = os.environ.get("LIVE_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
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
def _jl(x):
    return json.loads(x) if isinstance(x, str) else x


def _gamma(slug):
    """Gamma market record for a slug (tries the markets and events endpoints)."""
    for path, pick in (("markets", lambda js: js[0]), ("events", lambda js: js[0]["markets"][0])):
        try:
            js = S.get(f"https://gamma-api.polymarket.com/{path}", params={"slug": slug}, timeout=10).json()
            if js:
                return pick(js)
        except Exception:
            pass
    return None


def pm_market(H, start):
    """Up/Down token ids for the window starting at `start` (unix s), or None."""
    slug = f"btc-updown-{H}m-{start}"
    m = _gamma(slug)
    if not m:
        return None
    try:
        ids = _jl(m["clobTokenIds"])
        outs = [o.lower() for o in _jl(m["outcomes"])]
        return {"slug": slug, "up": ids[outs.index("up")], "down": ids[outs.index("down")]}
    except Exception:
        return None


def pm_outcome(slug):
    """1 if Polymarket resolved Up, 0 if Down, None if not resolved yet."""
    m = _gamma(slug)
    try:
        outs = [o.lower() for o in _jl(m["outcomes"])]
        px = [float(x) for x in _jl(m["outcomePrices"])]
        if max(px) > 0.99:
            return int(outs[px.index(max(px))] == "up")
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


def taker_fee(p):
    """Polymarket crypto-market taker fee per share: p * 0.25 * (p(1-p))^2 (max 1.56% at 0.50)."""
    return p * 0.25 * (p * (1 - p)) ** 2


# ---------------------------------------------------------------- fair value
def fair_up(prior, s0, st, sigma1m, secs_left, H):
    """P(close >= s0) given the current price st, blending the model prior as a drift tilt.

    The prior at window open implies a drift z0 = Phi^-1(prior) over the full window; the
    remaining drift scales with the fraction of time left.
    """
    if secs_left <= 0:
        return 1.0 if st >= s0 else 0.0
    tau = secs_left / 60.0
    z0 = _ppf(min(max(prior, 1e-4), 1 - 1e-4))
    sd = sigma1m * math.sqrt(tau)
    return _cdf((math.log(st / s0) / sd) + z0 * math.sqrt(tau / H))


def _cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _ppf(p):
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _cdf(mid) < p else (lo, mid)
    return (lo + hi) / 2


# ---------------------------------------------------------------- bookkeeping
def kelly(q, ask):
    """Kelly fraction of bankroll for buying a share at `ask` (+ taker fee) that pays 1 with
    probability q. Staking a quarter of this is the usual hedge against model error."""
    if ask is None:
        return None
    cost = ask + taker_fee(ask)
    return round(max(0.0, (q - cost) / (1 - cost)), 4) if cost < 1 else 0.0


class Book:
    """Predictions, open windows and running stats; persisted so restarts resume cleanly."""
    RECENT = 500  # windows in the rolling accuracy used for the drift warning

    def __init__(self):
        os.makedirs(os.path.join(LOG, "quotes"), exist_ok=True)
        self.pred_path = os.path.join(LOG, "live_predictions.csv")
        self.sum_path = os.path.join(LOG, "live_summary.json")
        self.open = {}  # (H, start) -> dict
        blank = {"n": 0, "correct": 0, "pm_n": 0, "pm_correct": 0, "trades": 0, "wins": 0, "pnl": 0.0,
                 "brier_sum": 0.0, "tiers": {}, "recent": []}
        self.stats = {str(H): json.loads(json.dumps(blank)) for H in (5, 15)}
        if os.path.exists(self.sum_path):
            js = json.load(open(self.sum_path))
            for H, st in js.get("stats", {}).items():
                self.stats[H].update({k: v for k, v in st.items() if k in blank})
            for k, w in js.get("open", {}).items():
                H, start = map(int, k.split(":"))
                w["fills"] = [tuple(f) for f in w.get("fills", [])]
                self.open[(H, start)] = w

    def quote_path(self):
        return os.path.join(LOG, "quotes", time.strftime("%Y-%m-%d.csv", time.gmtime()))

    def append(self, path, row):
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    def save(self):
        for s in self.stats.values():
            s["accuracy"] = round(s["correct"] / s["n"], 4) if s["n"] else None
            s["pm_accuracy"] = round(s["pm_correct"] / s["pm_n"], 4) if s["pm_n"] else None
            s["brier"] = round(s["brier_sum"] / s["n"], 5) if s["n"] else None
            s["pnl"] = round(s["pnl"], 4)
            s["tier_accuracy"] = {t: round(c / n, 4) for t, (n, c) in s["tiers"].items() if n}
            rec = s["recent"]
            s["recent_accuracy"] = round(sum(rec) / len(rec), 4) if rec else None
            # a coin flip over 300+ windows stays above 50 % less than half the time; below 49 % is a warning
            s["drift_warning"] = bool(len(rec) >= 300 and sum(rec) / len(rec) < 0.49)
        out = {"updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stats": self.stats,
               "open": {f"{H}:{t}": w for (H, t), w in self.open.items()}}
        tmp = self.sum_path + ".tmp"
        json.dump(out, open(tmp, "w"), indent=1)
        os.replace(tmp, self.sum_path)

    def resolve(self, d, now_t):
        """Close windows that have ended. Prefers Polymarket's own resolution (Chainlink) and
        waits up to 30 minutes for it; falls back to the exchange close."""
        for key in sorted(k for k in self.open if k[1] + 60 * k[0] <= now_t):
            H, start = key
            w = self.open[key]
            end_t = start + 60 * H
            end = d["c"][d["ts"] == end_t]
            if not len(end):
                if now_t - end_t > 3600:
                    self.open.pop(key)  # too old to resolve from the candle window
                continue
            pm = pm_outcome(w["mkt"]["slug"]) if w.get("mkt") else None
            if pm is None and w.get("mkt") and now_t - end_t < 1800:
                continue
            self.open.pop(key)
            ex_up = int(end[0] >= w["s0"])
            up = ex_up if pm is None else pm
            hit = int((w["p"] >= 0.5) == up)
            st = self.stats[str(H)]
            st["n"] += 1
            st["correct"] += int((w["p"] >= 0.5) == ex_up)
            st["brier_sum"] += (w["p"] - up) ** 2
            tn = st["tiers"].setdefault(w.get("tier", "?"), [0, 0])
            tn[0] += 1
            tn[1] += hit
            st["recent"] = (st["recent"] + [hit])[-self.RECENT:]
            if pm is not None:
                st["pm_n"] += 1
                st["pm_correct"] += int((w["p"] >= 0.5) == pm)
            for side, px in w.get("fills", []):
                win = (side == "up") == bool(up)
                st["trades"] += 1
                st["wins"] += int(win)
                st["pnl"] += (1.0 if win else 0.0) - px - taker_fee(px)
            self.append(self.pred_path, {
                "start_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime(start)), "start": start, "horizon": H,
                "feed": w["feed"], "s0": w["s0"], "s_end": float(end[0]), "p_up": round(w["p"], 4),
                "tier": w.get("tier", ""), "spread": round(w.get("spread", 0.0), 4),
                "exchange_up": ex_up, "polymarket_up": "" if pm is None else pm,
                "fills": json.dumps(w.get("fills", []))})


# ---------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-edge", type=float, default=0.03,
                    help="fair value - ask - taker fee needed to paper-buy one share")
    ap.add_argument("--poll", type=int, default=15, help="seconds between in-window quotes")
    ap.add_argument("--duration", type=float, default=0, help="exit cleanly after N minutes (0 = never)")
    ap.add_argument("--once", action="store_true", help="make one prediction and exit")
    args = ap.parse_args()
    model = Predictor()
    book = Book()
    last_minute = None
    t_end = time.time() + 60 * args.duration if args.duration else float("inf")
    log(f"started; models loaded; {len(book.open)} open windows restored")
    while time.time() < t_end:
        try:
            now = time.time()
            minute = int(now // 60 * 60)
            # once a minute: refresh candles and features (the partial-window model uses them)
            if minute != last_minute and now - minute >= 2:
                feed, d = candles(minute)
                if d["ts"][-1] != minute:
                    raise RuntimeError(f"{feed}: candle closing at {minute} not published yet")
                last_minute = minute  # only after a successful fetch, so failures retry
                p = model.predict(d)
                book.resolve(d, minute)
                if minute % 300 == 0:
                    s0 = float(d["c"][-1])
                    for H in (5, 15):
                        if minute % (60 * H):
                            continue
                        mkt = pm_market(H, minute)
                        q = p[H]
                        book.open[(H, minute)] = {"p": q["p_up"], "tier": q["tier"], "confidence": q["confidence"],
                                                  "spread": q["spread"], "s0": s0, "feed": feed,
                                                  "sigma": p["sigma1m"], "mkt": mkt, "fills": []}
                        log(f"{H:>2}m window {time.strftime('%H:%M', time.gmtime(minute))} "
                            f"P(up)={q['p_up']:.4f} -> {q['side'].upper()} conf={q['confidence']:.3f} "
                            f"tier={q['tier']} (backtest {q['tier_backtest_acc']:.1%}) spread={q['spread']:.3f} "
                            f"s0={s0:.2f} feed={feed} polymarket={mkt['slug'] if mkt else 'not found'}")
                book.save()
                if args.once and minute % 300 == 0:
                    return
            # in-window fair value vs the Polymarket book
            st_px = spot()
            for (H, start), w in list(book.open.items()):
                left = start + 60 * H - time.time()
                if st_px is None or not w["mkt"] or left <= 0:
                    continue
                elapsed = H - left / 60.0
                fu, src = None, "analytic"
                if elapsed >= 1:
                    fu = model.partial(math.log(st_px / w["s0"]), elapsed, H)
                    src = "partial_model"
                if fu is None:
                    fu, src = fair_up(w["p"], w["s0"], st_px, w["sigma"], left, H), "analytic"
                au, su = pm_ask(w["mkt"]["up"])
                ad, sd = pm_ask(w["mkt"]["down"])
                eu = None if au is None else round(fu - au - taker_fee(au), 4)
                ed = None if ad is None else round((1 - fu) - ad - taker_fee(ad), 4)
                book.append(book.quote_path(), {
                    "t": int(time.time()), "horizon": H, "start": start, "secs_left": int(left), "spot": st_px,
                    "fair_up": round(fu, 4), "source": src, "ask_up": au, "ask_up_size": su, "ask_down": ad,
                    "ask_down_size": sd, "edge_up": eu, "edge_down": ed,
                    "kelly_up": kelly(fu, au), "kelly_down": kelly(1 - fu, ad)})
                sides = {s for s, _ in w["fills"]}
                for side, edge, px in (("up", eu, au), ("down", ed, ad)):
                    if edge is not None and edge >= args.min_edge and side not in sides and left > 10:
                        w["fills"].append((side, px))
                        log(f"paper BUY {side} {H}m @ {px} fair={fu:.3f} ({src}) edge_after_fee={edge:.3f}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            log("error:", e)
            if "--debug" in sys.argv:
                traceback.print_exc()
            time.sleep(5)
        time.sleep(args.poll)
    book.save()
    log("stopped")


if __name__ == "__main__":
    main()
