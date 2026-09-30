"""Feature cache and time-based splits (no shuffling across time)."""
import os
import numpy as np
import pandas as pd
from . import data, features

CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "features.npz")

SPLITS = {  # [start, end) in UTC
    "pretrain": ("2014-03-01", "2023-01-01"),
    "finetune": ("2023-01-01", "2025-10-01"),
    "val": ("2025-10-01", "2026-04-01"),
    "test": ("2026-04-01", "2100-01-01"),
}
EMBARGO = 1440  # minutes dropped at each split edge so labels never straddle splits


def _t(s):
    return pd.Timestamp(s, tz="UTC").value // 10**9


def load(refresh=False):
    if not refresh and os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}
    d = data.load(refresh=refresh)
    X, names, ys, seq = features.build(d)
    out = dict(ts=d["ts"], X=X, names=np.array(names), seq=seq, c=d["c"].astype(np.float64),
               y5=ys[5], y15=ys[15], r5=ys["r5"], r15=ys["r15"])
    np.savez(CACHE, **out)
    return out


def split_idx(ts, name, horizon=None, stride=1):
    """Indices in a split. horizon=5/15 keeps only Polymarket-aligned window starts."""
    a, b = SPLITS[name]
    i = np.where((ts >= _t(a)) & (ts < _t(b)))[0]
    i = i[(i > i[0] + EMBARGO) & (i < i[-1] - EMBARGO)] if name != "test" else i[(i > i[0] + EMBARGO) & (i < len(ts) - 16)]
    i = i[i >= 1440 + features.SEQ_LEN]
    if horizon:
        i = i[ts[i] % (60 * horizon) == 0]
    return i[::stride]
