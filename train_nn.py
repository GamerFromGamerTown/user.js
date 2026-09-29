"""Pretrain on 2014-2022 BTC minutes, fine-tune on 2023-2025Q3, early-stop on val.

Saves models/nn.pt (weights + feature scaler). Evaluates on Polymarket-aligned test windows.
"""
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from btcpred import dataset as ds, features
from btcpred.model import Net
from btcpred.metrics import report

torch.set_num_threads(4)
torch.manual_seed(0)
np.random.seed(0)
os.makedirs("models", exist_ok=True)

d = ds.load()
ts, X, seq = d["ts"], d["X"], d["seq"]
Y = np.column_stack([d["y5"], d["y15"], d["r5"], d["r15"]]).astype(np.float32)
Y = np.nan_to_num(Y)
tie = np.column_stack([d["r5"] == 0, d["r15"] == 0])

pre = ds.split_idx(ts, "pretrain", stride=3)
mu, sd = X[pre].mean(0), X[pre].std(0) + 1e-6
Xn = ((X - mu) / sd).astype(np.float32)


def batch(idx):
    x = torch.from_numpy(Xn[idx])
    s = torch.from_numpy(features.windows(seq, idx))
    return x, s


def loss_fn(out, idx):
    y = torch.from_numpy(Y[idx])
    w = torch.from_numpy((~tie[idx]).astype(np.float32))
    bce = F.binary_cross_entropy_with_logits(out[:, :2], y[:, :2], weight=w, reduction="sum") / w.sum()
    aux = F.smooth_l1_loss(out[:, 2:], y[:, 2:])
    return bce + 0.2 * aux


@torch.no_grad()
def predict(net, idx, bs=16384):
    net.eval()
    ps = []
    for k in range(0, len(idx), bs):
        ps.append(torch.sigmoid(net(*batch(idx[k:k + bs]))[:, :2]).numpy())
    net.train()
    return np.concatenate(ps)


va5, va15 = ds.split_idx(ts, "val", 5), ds.split_idx(ts, "val", 15)
te5, te15 = ds.split_idx(ts, "test", 5), ds.split_idx(ts, "test", 15)


def val_score(net, quiet=True):
    a = report("val5", d["y5"][va5], predict(net, va5)[:, 0], quiet)
    b = report("val15", d["y15"][va15], predict(net, va15)[:, 1], quiet)
    return a["logloss"] + b["logloss"], a, b


def run(net, idx, epochs, lr, tag, bs=4096, patience=2):
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(idx) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        t0 = time.time()
        perm = np.random.permutation(idx)
        tot = 0.0
        for k in range(len(perm) // bs):
            b = np.sort(perm[k * bs:(k + 1) * bs])
            loss = loss_fn(net(*batch(b)), b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        sc, a, b_ = val_score(net)
        print(f"[{tag}] ep{ep} loss={tot / (k + 1):.5f} val_ll={sc:.5f} val5_acc={a['acc']:.4f} "
              f"val15_acc={b_['acc']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if sc < best:
            best, best_state, bad = sc, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net


if __name__ == "__main__":
    import sys
    pre_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    ft_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    net = Net(X.shape[1])
    pre = pre[~tie[pre].all(1)]
    print(f"pretrain n={len(pre)}", flush=True)
    if pre_epochs:
        net = run(net, pre, pre_epochs, 2e-3, "pretrain", patience=pre_epochs)
    zero = {k: v.clone() for k, v in net.state_dict().items()}
    ft = ds.split_idx(ts, "finetune")
    print(f"finetune n={len(ft)}", flush=True)
    net = run(net, ft, ft_epochs, 5e-4, "finetune")
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd, "n_feat": X.shape[1]}, "models/nn.pt")
    p5, p15 = predict(net, te5)[:, 0], predict(net, te15)[:, 1]
    report("test5 NN", d["y5"][te5], p5)
    report("test15 NN", d["y15"][te15], p15)
    np.savez("data/nn_test_preds.npz", te5=te5, te15=te15, p5=p5, p15=p15,
             v5=predict(net, va5)[:, 0], v15=predict(net, va15)[:, 1])
