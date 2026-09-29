"""Blend GBM and NN probabilities (weight picked on val log-loss), report the test backtest.

Writes models/ensemble.json (GBM weight per horizon) and logs/backtest.json.
"""
import json
import numpy as np
import lightgbm as lgb
from btcpred import dataset as ds
from btcpred.metrics import report

d = ds.load()
ts, X = d["ts"], d["X"]
nn = np.load("data/nn_test_preds.npz")
weights, results = {}, {}
for H in (5, 15):
    y, r = d[f"y{H}"], d[f"r{H}"]
    va, te = ds.split_idx(ts, "val", H), ds.split_idx(ts, "test", H)
    assert np.array_equal(te, nn[f"te{H}"])
    g = lgb.Booster(model_file=f"models/gbm_{H}.txt")
    gv, gt = g.predict(X[va]), g.predict(X[te])
    nv, nt = nn[f"v{H}"], nn[f"p{H}"]
    grid = np.linspace(0, 1, 21)
    ll = [report("", y[va], w * gv + (1 - w) * nv, quiet=True)["logloss"] for w in grid]
    w = float(grid[int(np.argmin(ll))])
    weights[str(H)] = w
    print(f"=== {H}m: GBM weight {w:.2f} (chosen on val)")
    p = w * gt + (1 - w) * nt
    report(f"test{H} GBM", y[te], gt)
    report(f"test{H} NN", y[te], nt)
    res = report(f"test{H} ENSEMBLE", y[te], p)
    keep = r[te] != 0  # Chainlink prices essentially never tie; Bitstamp ties resolve Up
    res_nt = report(f"test{H} ENSEMBLE no-ties", y[te][keep], p[keep])
    # paper PnL buying 1 share of the predicted side at 0.50 + half-spread (0.01), no fees
    hit = ((p >= 0.5) == (y[te] == 1)).astype(float)
    res["pnl_per_trade_at_0.51"] = float(hit.mean() - 0.51)
    results[str(H)] = {"all": res, "no_ties": res_nt, "gbm_weight": w,
                       "test_start": int(ts[te][0]), "test_end": int(ts[te][-1])}
json.dump(weights, open("models/ensemble.json", "w"))
json.dump(results, open("logs/backtest.json", "w"), indent=1, default=float)
