"""Features for predicting a window's outcome part-way through it ("partial chunks").

At elapsed minute k of an H-minute window that opened at index i0, the row is the regular
feature vector at i0+k plus the move so far measured against the time left.
"""
import math
import numpy as np

EXTRA = ["disp_z", "disp_bp", "elapsed", "remaining", "analytic_p"]


def _cdf(x):
    return 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


def extra(move, sigma1m, elapsed, H):
    """move: log(price_now / price_at_open); sigma1m: 1-minute log-return vol; elapsed in
    minutes (may be fractional). Returns the EXTRA columns as an (n, 5) float32 array."""
    move, sigma1m, elapsed = map(np.atleast_1d, (move, sigma1m, elapsed))
    rem = np.maximum(H - elapsed, 1e-3)
    z = move / (sigma1m * np.sqrt(rem))
    return np.column_stack([np.clip(z, -20, 20), move * 1e4, elapsed, rem, _cdf(z)]).astype(np.float32)


def rows(X, lc, sigma1m, i0, H):
    """All (window, elapsed minute) rows for windows opening at indices i0."""
    k = np.tile(np.arange(1, H), len(i0))
    start = np.repeat(i0, H - 1)
    i = start + k
    E = extra(lc[i] - lc[start], sigma1m[i], k.astype(float), H)
    return np.hstack([X[i], E]), start, k
