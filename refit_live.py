"""Refit the GBMs on 2023-01 -> latest data for live use (tree counts fixed by train_gbm.py).

Backtest numbers come from models/gbm_*.txt (trained only to 2025-09); these refits, which
also see the val/test period, go to models/live/ and are what live.py loads.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from btcpred import dataset as ds

os.makedirs("models/live", exist_ok=True)
d = ds.load()
ts, X = d["ts"], d["X"]
a = pd.Timestamp("2023-01-01", tz="UTC").value // 10**9
for H in (5, 15):
    y, r = d[f"y{H}"], d[f"r{H}"]
    tr = np.where((ts >= a) & (r != 0))[0][:-H][::2]
    base = lgb.Booster(model_file=f"models/gbm_{H}.txt")
    params = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=2000,
                  feature_fraction=0.7, bagging_fraction=0.5, bagging_freq=1, lambda_l2=10.0,
                  verbose=-1, num_threads=2)
    m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_boost_round=int(base.num_trees() * 1.2))
    m.save_model(f"models/live/gbm_{H}.txt")
    print(H, "refit on", len(tr), "rows,", m.num_trees(), "trees")
