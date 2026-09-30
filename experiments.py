"""GBM experiments, scored on the validation split only (the test split stays untouched).

Usage: python experiments.py [config ...]   (default: all configs)
"""
import sys
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from btcpred import dataset as ds

d = ds.load()
ts, X, names = d["ts"], d["X"], list(d["names"])
V1 = list(range(44))
BASE = dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=2000, feature_fraction=0.7,
            bagging_fraction=0.5, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4)


def t(s):
    return pd.Timestamp(s, tz="UTC").value // 10**9


def run(name, H, cols=None, target="bin", start=None, halflife_days=None, params=None, stride=2,
        aligned=False):
    y, r = d[f"y{H}"], d[f"r{H}"]
    tr = ds.split_idx(ts, "finetune", stride=stride)
    if start:
        tr = np.concatenate([np.where((ts >= t(start)) & (ts < t(ds.SPLITS["finetune"][0])))[0][::stride], tr])
    if aligned:
        tr = tr[ts[tr] % (60 * H) == 0]
    tr = tr[r[tr] != 0]
    va = ds.split_idx(ts, "val", H)
    cols = cols or list(range(X.shape[1]))
    w = None
    if target == "wbin":
        w = np.clip(np.abs(r[tr]), 0, 3)
    if halflife_days:
        age = (ts[tr].max() - ts[tr]) / 86400
        w = (w if w is not None else 1.0) * 0.5 ** (age / halflife_days)
    p = dict(BASE, **(params or {}))
    if target == "reg":
        p["objective"], yt, yv = "huber", r[tr], r[va]
    else:
        p["objective"], yt, yv = "binary", y[tr], y[va]
    t0 = time.time()
    m = lgb.train(p, lgb.Dataset(X[tr][:, cols], yt, weight=w), 3000,
                  valid_sets=[lgb.Dataset(X[va][:, cols], yv)],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    s = m.predict(X[va][:, cols], num_iteration=m.best_iteration)
    thr = 0.0 if target == "reg" else 0.5
    acc = np.mean((s >= thr) == (y[va] == 1))
    auc = roc_auc_score(y[va], s)
    ll = m.best_score["valid_0"].get("binary_logloss", float("nan"))
    print(f"{name:<22} H={H:<2} val_acc={acc:.4f} auc={auc:.4f} ll={ll:.5f} iters={m.best_iteration} "
          f"n={len(tr)} ({time.time() - t0:.0f}s)", flush=True)
    return m, acc


CONFIGS = {
    "v1_bin": dict(cols=V1),
    "v2_bin": dict(),
    "v2_reg": dict(target="reg"),
    "v2_wbin": dict(target="wbin"),
    "v2_2021_hl365": dict(start="2021-01-01", halflife_days=365),
    "v2_bin_big": dict(params=dict(num_leaves=63, min_data_in_leaf=5000, learning_rate=0.03)),
    "v2_bin_small": dict(params=dict(num_leaves=15, min_data_in_leaf=5000, feature_fraction=0.5)),
    "v2_2020_hl540_big": dict(start="2020-01-01", halflife_days=540,
                              params=dict(num_leaves=63, min_data_in_leaf=5000, learning_rate=0.03)),
    "v2_2019_hl365_s1": dict(start="2019-01-01", halflife_days=365, stride=1),
    "v2_reg_strong": dict(params=dict(num_leaves=15, min_data_in_leaf=10000, learning_rate=0.02,
                                      feature_fraction=0.5)),
    "v2_extratrees": dict(start="2021-01-01", halflife_days=365, params=dict(extra_trees=True)),
    "v2_aligned_2019": dict(start="2019-01-01", halflife_days=365, stride=1, aligned=True),
}

if __name__ == "__main__":
    for c in (sys.argv[1:] or CONFIGS):
        for H in (5, 15):
            run(c, H, **CONFIGS[c])
