"""Partial-window model: P(window closes Up) at each elapsed minute of a 5m / 15m window.

Trains on 2023-01 -> 2025-09 windows, early-stops on val, reports test accuracy, log-loss and
calibration per elapsed minute against the zero-drift analytic baseline Phi(move / (sigma*sqrt(t))).
Saves models/intra_{5,15}.txt.
"""
import json
import os
import numpy as np
import lightgbm as lgb
from btcpred import dataset as ds, intra
from btcpred.metrics import calibration

d = ds.load()
ts, X, names = d["ts"], d["X"], list(d["names"])
lc = np.log(d["c"])
sigma = np.exp(X[:, names.index("lvol60")].astype(np.float64))
params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=1000,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
              verbose=-1, num_threads=4)
summary = {}
for H in (5, 15):
    y, r = d[f"y{H}"], d[f"r{H}"]
    sets = {}
    for s in ("finetune", "val", "test"):
        i0 = ds.split_idx(ts, s, H)
        Xi, start, k = intra.rows(X, lc, sigma, i0, H)
        keep = r[start] != 0 if s == "finetune" else np.ones(len(start), bool)
        sets[s] = (Xi[keep], y[start][keep], k[keep])
    Xt, yt, _ = sets["finetune"]
    Xv, yv, _ = sets["val"]
    m = lgb.train(params, lgb.Dataset(Xt, yt), 3000, valid_sets=[lgb.Dataset(Xv, yv)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    m.save_model(f"models/intra_{H}.txt", num_iteration=m.best_iteration)
    Xs, ys, ks = sets["test"]
    p = m.predict(Xs, num_iteration=m.best_iteration)
    pa = Xs[:, -1]  # analytic baseline
    print(f"=== {H}m partial-window model, best_iter={m.best_iteration}")
    print(" elapsed  n      acc_model  acc_analytic  logloss_model  logloss_analytic  ECE_model")
    rows = []
    for kk in range(1, H):
        sel = ks == kk
        lm = -np.mean(ys[sel] * np.log(np.clip(p[sel], 1e-6, 1)) + (1 - ys[sel]) * np.log(np.clip(1 - p[sel], 1e-6, 1)))
        la = -np.mean(ys[sel] * np.log(np.clip(pa[sel], 1e-6, 1)) + (1 - ys[sel]) * np.log(np.clip(1 - pa[sel], 1e-6, 1)))
        am, aa = np.mean((p[sel] >= .5) == (ys[sel] == 1)), np.mean((pa[sel] >= .5) == (ys[sel] == 1))
        ece = calibration(ys[sel], p[sel])["ece"]
        print(f" {kk:>5}m  {sel.sum():<6} {am:.4f}     {aa:.4f}        {lm:.4f}         {la:.4f}          {ece:.4f}")
        rows.append({"elapsed": kk, "n": int(sel.sum()), "acc": am, "acc_analytic": aa,
                     "logloss": lm, "logloss_analytic": la, "ece": ece})
    summary[str(H)] = rows
    # refit on 2023-01 -> latest windows for live use
    i0 = ds.split_idx(ts, "finetune", H)
    i0 = np.concatenate([i0, np.where((ts >= ts[i0[-1]]) & (ts % (60 * H) == 0))[0]])
    i0 = np.unique(i0[i0 < len(ts) - H - 1])
    Xa, start, _ = intra.rows(X, lc, sigma, i0, H)
    keep = r[start] != 0
    ml = lgb.train(params, lgb.Dataset(Xa[keep], y[start][keep]), int(m.best_iteration * 1.1))
    os.makedirs("models/live", exist_ok=True)
    ml.save_model(f"models/live/intra_{H}.txt")
json.dump(summary, open("logs/intra_backtest.json", "w"), indent=1, default=float)
