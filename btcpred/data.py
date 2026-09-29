"""Load BTC/USD 1-minute candles into a gap-free numpy grid.

Source: https://github.com/ff137/bitstamp-btcusd-minute-data (Bitstamp, 2012 to present,
refreshed daily). Clone it next to this repo or pass the path via BTC_MINUTE_REPO.
"""
import os
import numpy as np
import pandas as pd

REPO = os.environ.get("BTC_MINUTE_REPO", "/home/user/ff137/bitstamp-btcusd-minute-data")
CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "btc_1m.npz")
START = pd.Timestamp("2014-01-01", tz="UTC").value // 10**9


def _read_csvs():
    hist = pd.read_csv(os.path.join(REPO, "data/historical/btcusd_bitstamp_1min_2012-2025.csv.gz"))
    upd = pd.read_csv(os.path.join(REPO, "data/updates/btcusd_bitstamp_1min_latest.csv"))
    df = pd.concat([hist, upd]).drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    return df[df.timestamp >= START]


def load(refresh=False):
    """Return dict of arrays on a 60 s grid: ts (minute close time), o, h, l, c, v.

    Bitstamp timestamps mark the candle open; ts here is open + 60, i.e. the moment the
    candle's close price is known. Missing minutes are forward-filled with zero volume.
    """
    if not refresh and os.path.exists(CACHE):
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    df = _read_csvs()
    t0, t1 = int(df.timestamp.iloc[0]), int(df.timestamp.iloc[-1])
    n = (t1 - t0) // 60 + 1
    idx = ((df.timestamp.values - t0) // 60).astype(np.int64)
    c = np.full(n, np.nan)
    c[idx] = df.close.values
    c = pd.Series(c).ffill().values
    out = {"ts": (t0 + 60 + 60 * np.arange(n)).astype(np.int64), "c": c}
    for k in ("open", "high", "low"):
        a = c.copy()
        a[idx] = df[k].values
        out[k[0]] = a
    v = np.zeros(n)
    v[idx] = df.volume.values
    out["v"] = v
    for k in out:
        if k != "ts":
            out[k] = out[k].astype(np.float64)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, **out)
    return out


if __name__ == "__main__":
    d = load(refresh=True)
    print(len(d["ts"]), pd.to_datetime(d["ts"][[0, -1]], unit="s"))
