"""Causal features and Polymarket-style labels on the 1-minute grid.

Index i is the moment ts[i]; every feature at i uses only candles closed at or before ts[i].
Label y{H}[i] = 1 if close[i+H] >= close[i] (Polymarket "Up" resolves on >=).
"""
import numpy as np
import pandas as pd

RET_LAGS = [1, 2, 3, 5, 10, 15, 30, 60, 120, 240, 480, 1440]
SEQ_LEN = 64


def _roll(x, w, fn):
    return getattr(pd.Series(x).rolling(w, min_periods=w), fn)().values


def build(d):
    ts, c, h, l, v = d["ts"], d["c"], d["h"], d["l"], d["v"]
    lc, lh, ll = np.log(c), np.log(h), np.log(l)
    r1 = np.diff(lc, prepend=lc[0])
    vol = {w: np.maximum(_roll(r1, w, "std"), 1e-5) for w in (15, 60, 240, 1440)}
    s = vol[60]
    f = {}
    for k in RET_LAGS:
        rk = lc - np.roll(lc, k)
        f[f"z{k}"] = rk / (s * np.sqrt(k))
    for w in (15, 60, 240, 1440):
        f[f"lvol{w}"] = np.log(vol[w])
    f["vr15_240"] = np.log(vol[15] / vol[240])
    f["vr60_1440"] = np.log(vol[60] / vol[1440])
    for k in (5, 15, 60, 240):
        hi, lo = _roll(lh, k, "max"), _roll(ll, k, "min")
        rng = np.maximum(hi - lo, 1e-9)
        f[f"rng{k}"] = rng / (s * np.sqrt(k))
        f[f"pos{k}"] = (lc - lo) / rng
    lv = np.log1p(v)
    base = _roll(v, 1440, "mean") + 1e-9
    sv = v * np.sign(r1)
    for k in (5, 15, 60):
        f[f"v{k}"] = np.log(_roll(v, k, "mean") / base + 1e-6)
        f[f"sv{k}"] = _roll(sv, k, "sum") / (_roll(v, k, "sum") + 1e-9)
    for span in (10, 30, 120, 480):
        ema = pd.Series(lc).ewm(span=span, adjust=False).mean().values
        f[f"ema{span}"] = (lc - ema) / (s * np.sqrt(span))
    body = (c - d["o"]) / np.maximum(h - l, 1e-9)
    f["body1"] = body
    f["body5"] = _roll(body, 5, "mean")
    f["notrade"] = (v == 0).astype(float)
    tod = (ts % 86400) / 86400.0
    dow = ((ts // 86400) + 4) % 7 / 7.0  # 1970-01-01 was a Thursday
    f["tod_s"], f["tod_c"] = np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod)
    f["dow_s"], f["dow_c"] = np.sin(2 * np.pi * dow), np.cos(2 * np.pi * dow)
    f["m15"] = (ts % 900) / 900.0  # position inside the 15-minute block
    X = np.column_stack([np.nan_to_num(f[k], nan=0.0, posinf=0.0, neginf=0.0) for k in f]).astype(np.float32)
    X = np.clip(X, -20, 20)
    ys = {}
    for H in (5, 15):
        fut = np.roll(lc, -H)
        y = (fut >= lc).astype(np.float32)
        y[-H:] = np.nan
        ys[H] = y
        # future return in vol units, for an auxiliary regression target
        rr = (fut - lc) / (s * np.sqrt(H))
        rr[-H:] = 0.0
        ys[f"r{H}"] = np.clip(rr, -10, 10).astype(np.float32)
    # per-minute sequence channels (normalised return, range, log-volume z) for the NN
    seq = np.column_stack([
        np.clip(r1 / s, -10, 10),
        np.clip((lh - ll) / s, 0, 20),
        np.clip((lv - _roll(lv, 1440, "mean")) / (_roll(lv, 1440, "std") + 1e-6), -5, 5),
    ])
    seq = np.nan_to_num(seq).astype(np.float32)
    return X, list(f), ys, seq


def windows(seq, idx, L=SEQ_LEN):
    """Stack seq[i-L+1 .. i] for each i in idx -> (n, L, C)."""
    off = np.arange(-L + 1, 1)
    return seq[idx[:, None] + off[None, :]]
