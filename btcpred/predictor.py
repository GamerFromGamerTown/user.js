"""Load trained models and score the latest candle window.

predict()  -> calibrated P(Up) for the 5m / 15m windows opening now, with confidence tier
partial()  -> P(Up) for a window already in progress, from the move so far and time left
"""
import json
import os
import numpy as np
import lightgbm as lgb
import torch
from scipy.special import expit, logit
from . import features, intra
from .model import Net

MODELS = os.path.join(os.path.dirname(__file__), "..", "models")


def _prefer_live(models_dir, rel):
    live = os.path.join(models_dir, "live", rel)
    return live if os.path.exists(live) else os.path.join(models_dir, rel)


class Predictor:
    def __init__(self, models_dir=MODELS):
        self.cfg = json.load(open(os.path.join(models_dir, "ensemble.json")))
        self.gbm = {H: {name: lgb.Booster(model_file=_prefer_live(models_dir, f"gbm/{H}_{name}.txt"))
                        for name in self.cfg["members"]} for H in (5, 15)}
        ck = torch.load(os.path.join(models_dir, "nn.pt"), weights_only=False)
        self.nn = Net(ck["n_feat"])
        self.nn.load_state_dict(ck["state"])
        self.nn.eval()
        self.mu, self.sd = ck["mu"], ck["sd"]
        self.intra = {}
        for H in (5, 15):
            path = _prefer_live(models_dir, f"intra_{H}.txt")
            if os.path.exists(path):
                self.intra[H] = lgb.Booster(model_file=path)
        self.x_last = self.sigma_last = None

    def predict(self, d):
        """d: gap-free 1-minute arrays (ts, o, h, l, c, v), >= 3000 rows, last row = latest
        closed candle. Returns {5: info, 15: info, "sigma1m": float}, where info holds
        p_up (calibrated), side, confidence = P(predicted side), tier, the tier's backtest
        accuracy and spread (std of the member models' probabilities: disagreement)."""
        X, names, _, seq = features.build(d)
        i = np.array([len(X) - 1])
        self.x_last = X[-1]
        self.sigma_last = float(np.exp(X[-1, names.index("lvol60")]))
        x = torch.from_numpy(((X[i] - self.mu) / self.sd).astype(np.float32))
        with torch.no_grad():
            pn = torch.sigmoid(self.nn(x, torch.from_numpy(features.windows(seq, i)))[0, :2]).numpy()
        out = {"sigma1m": self.sigma_last}
        for k, H in enumerate((5, 15)):
            c = self.cfg[str(H)]
            probs = []
            for name, m in self.gbm[H].items():
                s = float(m.predict(X[i][:, :self.cfg["n_v1"]] if name == "v1" else X[i])[0])
                probs.append(float(expit(2.0 * s)) if name == "reg" else s)
            zg = np.mean(logit(np.clip(probs, 1e-4, 1 - 1e-4)))
            zn = float(logit(np.clip(pn[k], 1e-4, 1 - 1e-4)))
            w, (a, b) = c["nn_weight"], c["platt"]
            p = float(expit(a * ((1 - w) * zg + w * zn) + b))
            conf = abs(p - 0.5)
            tier = next((t for t in c["tiers"] if conf >= t["min_conf"]), c["tiers"][-1])
            out[H] = {"p_up": p, "side": "up" if p >= 0.5 else "down", "confidence": max(p, 1 - p),
                      "tier": tier["tier"], "tier_backtest_acc": tier["test_acc"],
                      "spread": float(np.std(probs + [float(pn[k])]))}
        return out

    def partial(self, move, elapsed_min, H):
        """P(Up) for an H-minute window `elapsed_min` minutes in, where move = log(spot /
        price at the window open). Uses the features from the last predict() call."""
        if H not in self.intra or self.x_last is None:
            return None
        row = np.hstack([self.x_last[None, :], intra.extra(move, self.sigma_last, elapsed_min, H)])
        return float(self.intra[H].predict(row)[0])
