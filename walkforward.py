"""Walk-forward check: retrain the GBM on an earlier window and score later, unseen months.

Uses the tree counts found by train_final.py, so no future data picks the model size.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from btcpred import dataset as ds
from btcpred.metrics import report

d = ds.load()
ts, X = d["ts"], d["X"]
t = lambda s: pd.Timestamp(s, tz="UTC").value // 10**9
FOLDS = [("2021-01-01", "2023-10-01", "2023-10-02", "2024-04-01"),
         ("2021-07-01", "2024-04-01", "2024-04-02", "2024-10-01"),
         ("2022-01-01", "2024-10-01", "2024-10-02", "2025-04-01"),
         ("2022-07-01", "2025-04-01", "2025-04-02", "2025-10-01")]
for H in (5, 15):
    y, r = d[f"y{H}"], d[f"r{H}"]
    rounds = lgb.Booster(model_file=f"models/gbm/{H}_v1.txt").num_trees()
    params = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=2000,
                  feature_fraction=0.7, bagging_fraction=0.5, bagging_freq=1, lambda_l2=10.0,
                  verbose=-1, num_threads=2)
    for a, b, c, e in FOLDS:
        tr = np.where((ts >= t(a)) & (ts < t(b)) & (r != 0))[0][::2]
        te = np.where((ts >= t(c)) & (ts < t(e)) & (ts % (60 * H) == 0))[0]
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_boost_round=rounds)
        report(f"{H}m test {c[:7]}..{e[:7]}", y[te], m.predict(X[te]))
