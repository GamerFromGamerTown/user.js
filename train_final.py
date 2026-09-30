"""Final pre-window ensemble: 5 GBM variants + NN, calibrated, with confidence tiers.

Everything is chosen on the validation split; the test split is scored once at the end.
Writes models/gbm/*.txt (backtest models), models/live/gbm/*.txt (refit to the latest data),
models/ensemble.json (members, NN weight, calibration, tiers) and logs/backtest.json.
Run train_nn.py first (it writes data/nn_test_preds.npz and models/nn.pt).
"""
import json
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from btcpred import dataset as ds
from btcpred.metrics import report, calibration

d = ds.load()
ts, X = d["ts"], d["X"]
N_V1 = 44
BASE = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=2000,
            feature_fraction=0.7, bagging_fraction=0.5, bagging_freq=1, lambda_l2=10.0,
            verbose=-1, num_threads=4)
# name: (feature count, train start, half-life days, stride, extra params)
MEMBERS = {
    "v1": (N_V1, None, None, 2, {}),
    "big": (None, None, None, 2, dict(num_leaves=63, min_data_in_leaf=5000, learning_rate=0.03)),
    "xt": (None, "2021-01-01", 365, 2, dict(extra_trees=True)),
    "long": (None, "2019-01-01", 365, 1, {}),
    "reg": (None, None, None, 2, dict(objective="huber")),
}
TIERS = [("high", 0.10), ("medium", 0.30), ("low", 0.60)]  # top 10 %, next 20 %, next 30 % by |p-0.5|


def t(s):
    return pd.Timestamp(s, tz="UTC").value // 10**9


def train_idx(start, stride, end=None):
    a = t(start) if start else t(ds.SPLITS["finetune"][0])
    b = t(end) if end else t(ds.SPLITS["finetune"][1])
    i = np.where((ts >= a) & (ts < b))[0][::stride]
    return i[i >= 1440 + 64]


def weights(i, hl):
    if not hl:
        return None
    return 0.5 ** ((ts[i].max() - ts[i]) / 86400 / hl)


def member_prob(m, name, Xs):
    s = m.predict(Xs)
    if name == "reg":  # vol-normalised return forecast -> probability via a fixed logistic link
        s = expit(s * 2.0)
    return s


os.makedirs("models/gbm", exist_ok=True)
os.makedirs("models/live/gbm", exist_ok=True)
nn = np.load("data/nn_test_preds.npz")
cfg, results = {"members": list(MEMBERS), "n_v1": N_V1}, {}
for H in (5, 15):
    y, r = d[f"y{H}"], d[f"r{H}"]
    va, te = ds.split_idx(ts, "val", H), ds.split_idx(ts, "test", H)
    zv, zt, iters = [], [], {}
    for name, (nf, start, hl, stride, extra) in MEMBERS.items():
        cols = slice(0, nf) if nf else slice(None)
        tr = train_idx(start, stride)
        tr = tr[r[tr] != 0]
        p = dict(BASE, **extra)
        yt, yv = (r[tr], r[va]) if name == "reg" else (y[tr], y[va])
        m = lgb.train(p, lgb.Dataset(X[tr][:, cols], yt, weight=weights(tr, hl)), 3000,
                      valid_sets=[lgb.Dataset(X[va][:, cols], yv)],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        m.save_model(f"models/gbm/{H}_{name}.txt", num_iteration=m.best_iteration)
        iters[name] = m.best_iteration
        zv.append(logit(np.clip(member_prob(m, name, X[va][:, cols]), 1e-4, 1 - 1e-4)))
        zt.append(logit(np.clip(member_prob(m, name, X[te][:, cols]), 1e-4, 1 - 1e-4)))
        print(f"{H}m member {name}: iters={m.best_iteration}", flush=True)
    gv, gt = np.mean(zv, 0), np.mean(zt, 0)
    assert np.array_equal(te, nn[f"te{H}"])
    nv = logit(np.clip(nn[f"v{H}"], 1e-4, 1 - 1e-4))
    nt = logit(np.clip(nn[f"p{H}"], 1e-4, 1 - 1e-4))
    # NN weight and Platt calibration, both fit on validation
    best = None
    for w in np.linspace(0, 1, 21):
        lr = LogisticRegression(C=1e6).fit(((1 - w) * gv + w * nv)[:, None], y[va])
        pv = lr.predict_proba(((1 - w) * gv + w * nv)[:, None])[:, 1]
        ll = -np.mean(y[va] * np.log(pv) + (1 - y[va]) * np.log(1 - pv))
        if best is None or ll < best[0]:
            best = (ll, w, float(lr.coef_[0, 0]), float(lr.intercept_[0]))
    _, w, a, b = best
    pv = expit(a * ((1 - w) * gv + w * nv) + b)
    pt = expit(a * ((1 - w) * gt + w * nt) + b)
    conf_v = np.abs(pv - 0.5)
    thr = [float(np.quantile(conf_v, 1 - q)) for _, q in TIERS]
    print(f"=== {H}m: NN weight {w:.2f}, Platt a={a:.3f} b={b:.4f}")
    report(f"val{H} ensemble", y[va], pv)
    res = report(f"test{H} ensemble", y[te], pt)
    res["ece"] = calibration(y[te], pt)["ece"]
    res["brier"] = float(np.mean((pt - y[te]) ** 2))
    tiers = []
    conf_t = np.abs(pt - 0.5)
    lo_prev = np.inf
    for (tname, _), lo in zip(TIERS, thr):
        sel = (conf_t >= lo) & (conf_t < lo_prev)
        selv = (conf_v >= lo) & (conf_v < lo_prev)
        tiers.append({"tier": tname, "min_conf": lo,
                      "val_acc": float(np.mean((pv[selv] >= .5) == (y[va][selv] == 1))),
                      "test_acc": float(np.mean((pt[sel] >= .5) == (y[te][sel] == 1))),
                      "test_share": float(sel.mean())})
        lo_prev = lo
    sel = conf_t < lo_prev
    tiers.append({"tier": "none", "min_conf": 0.0,
                  "test_acc": float(np.mean((pt[sel] >= .5) == (y[te][sel] == 1))), "test_share": float(sel.mean())})
    for tr_ in tiers:
        print(f"   tier {tr_['tier']:<7} share={tr_['test_share']:.2f} test_acc={tr_['test_acc']:.4f}")
    # accuracy vs coverage: trade only the most confident fraction of windows
    cover = {}
    for q in (1.0, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05):
        k = conf_t >= np.quantile(conf_t, 1 - q)
        cover[str(q)] = float(np.mean((pt[k] >= .5) == (y[te][k] == 1)))
    print("   coverage->acc:", {k: round(v, 4) for k, v in cover.items()})
    res.update({"tiers": tiers, "coverage_acc": cover, "nn_weight": w,
                "test_start": int(ts[te][0]), "test_end": int(ts[te][-1])})
    results[str(H)] = res
    cfg[str(H)] = {"nn_weight": w, "platt": [a, b], "tiers": tiers, "iters": iters}
    # refit members on everything from their start date to the latest data, for live use
    for name, (nf, start, hl, stride, extra) in MEMBERS.items():
        cols = slice(0, nf) if nf else slice(None)
        tr = train_idx(start, stride, end="2100-01-01")
        tr = tr[(r[tr] != 0) & (tr < len(ts) - H)]
        p = dict(BASE, **extra)
        yt = r[tr] if name == "reg" else y[tr]
        m = lgb.train(p, lgb.Dataset(X[tr][:, cols], yt, weight=weights(tr, hl)), int(iters[name] * 1.2))
        m.save_model(f"models/live/gbm/{H}_{name}.txt")
json.dump(cfg, open("models/ensemble.json", "w"), indent=1)
json.dump(results, open("logs/backtest.json", "w"), indent=1, default=float)
