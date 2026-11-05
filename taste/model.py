"""Taste model: learns your ratings from candidate features.

Few ratings  -> ridge regression (stable with little data, explains itself).
>= NN_MIN    -> a small neural network (numpy MLP), used only if it beats ridge
                in cross-validation.
"""
import json
import math
import random

import numpy as np

from .features import NAMES, DESCRIPTIONS

MIN_RATINGS = 8
NN_MIN = 100


def _standardize(X):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-9] = 1.0
    return mu, sd


# ---------------- ridge ----------------
def ridge_fit(X, y, lam):
    n, d = X.shape
    A = X.T @ X + lam * np.eye(d)
    w = np.linalg.solve(A, X.T @ (y - y.mean()))
    return w, y.mean()


def _cv_error(fit, predict, X, y, folds=5, seed=0):
    idx = list(range(len(y)))
    random.Random(seed).shuffle(idx)
    k = min(folds, len(y))
    err = []
    for f in range(k):
        test = idx[f::k]
        train = [i for i in idx if i not in test]
        if len(train) < 3:
            continue
        m = fit(X[train], y[train])
        err += [(predict(m, X[i]) - y[i]) ** 2 for i in test]
    return math.sqrt(sum(err) / len(err)) if err else float('inf')


# ---------------- tiny MLP ----------------
def mlp_fit(X, y, hidden=8, epochs=3000, lr=0.01, l2=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    W1 = rng.normal(0, 0.3, (d, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.3, hidden); b2 = y.mean()
    for _ in range(epochs):
        h = np.tanh(X @ W1 + b1)
        p = h @ W2 + b2
        g = 2 * (p - y) / len(y)
        gW2 = h.T @ g + l2 * W2
        gb2 = g.sum()
        gh = np.outer(g, W2) * (1 - h ** 2)
        gW1 = X.T @ gh + l2 * W1
        gb1 = gh.sum(0)
        W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2
    return dict(W1=W1.tolist(), b1=b1.tolist(), W2=W2.tolist(), b2=float(b2))


def mlp_predict(m, x):
    h = np.tanh(np.asarray(x) @ np.asarray(m['W1']) + np.asarray(m['b1']))
    return float(h @ np.asarray(m['W2']) + m['b2'])


# ---------------- public API ----------------
def train(X, y):
    X = np.asarray(X, float); y = np.asarray(y, float)
    if len(y) < MIN_RATINGS:
        raise ValueError(f'need at least {MIN_RATINGS} ratings, have {len(y)}')
    mu, sd = _standardize(X)
    Z = (X - mu) / sd

    best = None
    for lam in (0.3, 1, 3, 10, 30, 100):
        fit = lambda A, b, lam=lam: ridge_fit(A, b, lam)
        pred = lambda m, x: float(x @ m[0] + m[1])
        e = _cv_error(fit, pred, Z, y)
        if best is None or e < best[0]:
            best = (e, lam)
    w, b = ridge_fit(Z, y, best[1])
    model = dict(kind='ridge', lam=best[1], w=w.tolist(), b=float(b), cv_rmse=best[0])

    baseline = math.sqrt(((y - y.mean()) ** 2).mean())
    model['baseline_rmse'] = baseline

    if len(y) >= NN_MIN:
        e_nn = _cv_error(lambda A, b: mlp_fit(A, b), mlp_predict, Z, y)
        if e_nn < best[0]:
            model.update(kind='mlp', mlp=mlp_fit(Z, y), cv_rmse=e_nn)
    model.update(mu=mu.tolist(), sd=sd.tolist(), names=NAMES, n=len(y),
                 ridge_w=w.tolist(), ridge_b=float(b))
    return model


def predict(model, x):
    z = (np.asarray(x, float) - np.asarray(model['mu'])) / np.asarray(model['sd'])
    if model['kind'] == 'mlp':
        return mlp_predict(model['mlp'], z)
    return float(z @ np.asarray(model['w']) + model['b'])


def explain(model, top=6):
    """Which features push your rating up / down (from the ridge weights)."""
    pairs = sorted(zip(model['ridge_w'], model['names']), reverse=True)
    likes = [(w, n) for w, n in pairs if w > 0][:top]
    dislikes = [(w, n) for w, n in reversed(pairs) if w < 0][:top]
    fmt = lambda w, n: f'  {w:+.2f}  {DESCRIPTIONS.get(n, n)}'
    lines = ['you tend to like more:'] + [fmt(w, n) for w, n in likes]
    lines += ['you tend to like less:'] + [fmt(w, n) for w, n in dislikes]
    return '\n'.join(lines)


def save(model, path):
    with open(path, 'w') as f:
        json.dump(model, f, indent=1)


def load(path):
    with open(path) as f:
        return json.load(f)
