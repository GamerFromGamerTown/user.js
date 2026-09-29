import numpy as np


def report(tag, y, p, quiet=False):
    """Accuracy overall and on the most confident fractions (where a bettor would act)."""
    y, p = np.asarray(y), np.asarray(p)
    acc = np.mean((p >= 0.5) == (y == 1))
    se = np.sqrt(0.25 / len(y))
    ll = -np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
    conf = np.abs(p - 0.5)
    out = {"n": len(y), "acc": acc, "se": se, "logloss": ll, "base_up": y.mean()}
    parts = []
    for q in (0.5, 0.2, 0.1):
        k = conf >= np.quantile(conf, 1 - q)
        a = np.mean((p[k] >= 0.5) == (y[k] == 1))
        out[f"acc_top{int(q*100)}"] = a
        parts.append(f"top{int(q*100)}%={a:.4f}(n={k.sum()})")
    if not quiet:
        print(f"[{tag}] n={len(y)} acc={acc:.4f} (±{se:.4f} 1SE) logloss={ll:.5f} "
              f"base_up={y.mean():.4f} | " + " ".join(parts))
    return out
