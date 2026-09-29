"""Load trained models and score the latest candle window."""
import json
import os
import numpy as np
import lightgbm as lgb
import torch
from . import features
from .model import Net

MODELS = os.path.join(os.path.dirname(__file__), "..", "models")


class Predictor:
    def __init__(self, models_dir=MODELS):
        # prefer the GBMs refit on data up to the latest day (refit_live.py)
        gdir = os.path.join(models_dir, "live") if os.path.exists(os.path.join(models_dir, "live", "gbm_5.txt")) else models_dir
        self.gbm = {H: lgb.Booster(model_file=os.path.join(gdir, f"gbm_{H}.txt")) for H in (5, 15)}
        cfg_path = os.path.join(models_dir, "ensemble.json")
        self.w = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {"5": 1.0, "15": 1.0}
        self.nn = None
        nn_path = os.path.join(models_dir, "nn.pt")
        if os.path.exists(nn_path):
            ck = torch.load(nn_path, weights_only=False)
            self.nn = Net(ck["n_feat"])
            self.nn.load_state_dict(ck["state"])
            self.nn.eval()
            self.mu, self.sd = ck["mu"], ck["sd"]

    def predict(self, d):
        """d: gap-free 1-minute arrays (ts, o, h, l, c, v), >= 3000 rows, last row = latest
        closed candle. Returns {5: P(up over next 5m), 15: P(up over next 15m)}."""
        X, _, _, seq = features.build(d)
        i = np.array([len(X) - 1])
        out = {}
        pn = None
        if self.nn is not None:
            x = torch.from_numpy(((X[i] - self.mu) / self.sd).astype(np.float32))
            s = torch.from_numpy(features.windows(seq, i))
            with torch.no_grad():
                pn = torch.sigmoid(self.nn(x, s)[0, :2]).numpy()
        for k, H in enumerate((5, 15)):
            pg = float(self.gbm[H].predict(X[i])[0])
            wg = float(self.w.get(str(H), 1.0))
            out[H] = pg if pn is None else wg * pg + (1 - wg) * float(pn[k])
        # recent 1-minute log-return volatility, used for the intra-window fair value
        r = np.diff(np.log(d["c"][-61:]))
        out["sigma1m"] = float(max(r.std(), 1e-5))
        return out
