"""Gradient-boosted baseline on the handcrafted features. Saves models/gbm_{5,15}.txt."""
import os
import sys
import numpy as np
import lightgbm as lgb
from btcpred import dataset as ds
from btcpred.metrics import report

os.makedirs("models", exist_ok=True)
d = ds.load()
ts, X = d["ts"], d["X"]
train_from = sys.argv[1] if len(sys.argv) > 1 else "finetune"
tag = sys.argv[2] if len(sys.argv) > 2 else ""

for H in (5, 15):
    y, r = d[f"y{H}"], d[f"r{H}"]
    tr = np.concatenate([ds.split_idx(ts, s, stride=2) for s in train_from.split(",")])
    tr = tr[r[tr] != 0]  # drop exact ties (mostly no-trade minutes in thin years)
    va, te = ds.split_idx(ts, "val", H), ds.split_idx(ts, "test", H)
    params = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=2000,
                  feature_fraction=0.7, bagging_fraction=0.5, bagging_freq=1, lambda_l2=10.0,
                  verbose=-1, num_threads=4)
    m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_boost_round=2000,
                  valid_sets=[lgb.Dataset(X[va], y[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    m.save_model(f"models/gbm{tag}_{H}.txt")
    print(f"--- GBM {H}m  train={train_from} n_train={len(tr)} best_iter={m.best_iteration}")
    report("val", y[va], m.predict(X[va], num_iteration=m.best_iteration))
    report("test", y[te], m.predict(X[te], num_iteration=m.best_iteration))
    imp = sorted(zip(m.feature_importance("gain"), d["names"]), reverse=True)[:10]
    print("top features:", ", ".join(n for _, n in imp))
