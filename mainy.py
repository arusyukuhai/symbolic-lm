# =========================================================
# GP-D3PM (Discrete Diffusion) over Tiny Shakespeare (char-level)
#  - NO NN, NO n-gram: only GP program graph + your op set
#  - Reverse model: p_theta(x_{t-1} | x_t, t)
#  - Output: last_k = vocab_size nodes => logits per vocab char, per position
# =========================================================

import os
import time
import gc
import math
import urllib.request
from copy import deepcopy

import numpy as np
from numba import njit
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -------------------------
# Tiny Shakespeare download
# -------------------------
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def load_tiny_shakespeare(path="input.txt"):
    if not os.path.exists(path):
        print("डाउनलोड: Tiny Shakespeare ...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for ch,i in stoi.items()}
    return chars, stoi, itos

def encode(text, stoi):
    return np.array([stoi[c] for c in text], dtype=np.int32)

def decode(ids, itos):
    return "".join(itos[int(i)] for i in ids)

def split_train_val_test(text, train_ratio=0.9, val_ratio=0.05):
    n = len(text)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = text[:n_train]
    val = text[n_train:n_train+n_val]
    test = text[n_train+n_val:]
    return train, val, test

# =========================================================
# Your helper + function sets (ほぼ踏襲)
# =========================================================

# 1D helper (TT/TT2)
def TT(x):
    if x.size < 3:
        return x.copy()
    return np.concatenate((np.ones(1, x.dtype),
                           (x[:-2] + x[1:-1]*2 + x[2:]) * 0.25,
                           np.ones(1, x.dtype)))

def TT2(x):
    if x.size < 3:
        return x.copy()
    return np.concatenate((np.ones(1, x.dtype),
                           ((x[:-2] - x[1:-1])**2 + (x[1:-1] - x[2:])**2) * 0.5,
                           np.ones(1, x.dtype)))


import numpy as np

def _attn_poly_fast(k, v, q, deg: int):
    # --- dtype & sanitize ---
    is_cplx = np.iscomplexobj(k) or np.iscomplexobj(v) or np.iscomplexobj(q)
    dtype = np.complex128 if is_cplx else np.float64

    k = np.asarray(k, dtype=dtype).ravel()
    v = np.asarray(v, dtype=dtype).ravel()
    q = np.asarray(q, dtype=dtype).ravel()

    if dtype == np.float64:
        k = np.nan_to_num(k, nan=0.0, posinf=1e6, neginf=-1e6)
        v = np.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)
        q = np.nan_to_num(q, nan=0.0, posinf=1e6, neginf=-1e6)
    else:
        k = np.where(np.isfinite(k.real) & np.isfinite(k.imag), k, 0.0 + 0.0j)
        v = np.where(np.isfinite(v.real) & np.isfinite(v.imag), v, 0.0 + 0.0j)
        q = np.where(np.isfinite(q.real) & np.isfinite(q.imag), q, 0.0 + 0.0j)

    n = k.size
    if n == 0:
        return np.zeros_like(q, dtype=dtype)

    deg = int(deg)
    if deg < 0:
        return np.zeros_like(q, dtype=dtype)

    size = deg + 1

    # --- normalize ---
    scale = np.max(np.abs(k)) + 1e-12
    kn = k / scale
    qn = q / scale

    # --- build Vandermonde powers (n x size) ---
    # X[t,i] = kn[t]^i
    X = np.empty((n, size), dtype=dtype)
    X[:, 0] = 1
    for i in range(1, size):
        X[:, i] = X[:, i - 1] * kn

    # --- Hermitian normal equations for complex LS ---
    # A = Xᴴ X, b = Xᴴ v
    A = X.conj().T @ X
    b = X.conj().T @ v

    # --- ridge ---
    tr = np.trace(A)
    ridge = 1e-8 * (tr.real / size) if size > 0 else 1e-10
    A = A.copy()
    A.flat[::size + 1] += max(ridge, 1e-10)

    # --- solve (Cholesky -> fallback) ---
    try:
        L = np.linalg.cholesky(A)
        y = np.linalg.solve(L, b)
        coef = np.linalg.solve(L.conj().T, y)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(A, b, rcond=1e-15)[0]

    # --- evaluate with Horner (avoid Q matrix) ---
    out = np.zeros_like(qn, dtype=dtype)
    for c in coef[::-1]:
        out = out * qn + c
    return out

def attn_poly3_fast(k, v, q):  return _attn_poly_fast(np.nan_to_num(k), np.nan_to_num(v), np.nan_to_num(q), deg=3)
def attn_poly5_fast(k, v, q):  return _attn_poly_fast(np.nan_to_num(k), np.nan_to_num(v), np.nan_to_num(q), deg=5)
def attn_poly11_fast(k, v, q): return _attn_poly_fast(np.nan_to_num(k), np.nan_to_num(v), np.nan_to_num(q), deg=11)

gg   = lambda x: np.abs(x)**(1/3)  * np.sign(x)
ggg  = lambda x: np.abs(x)**0.2    * np.sign(x)
gggg = lambda x: np.abs(x)**(1/11) * np.sign(x)

# 1/2/3-ary function sets
funcs_1 = [
    lambda x: x + 1,
    lambda x: x * 2,
    lambda x: x ** 2,
    lambda x: -x,
    lambda x: 1 / (x ** 2 + 1e-12),
    lambda x: np.sort(x),
    lambda x: np.argsort(x).astype(np.float32),
    lambda x: np.argsort(np.argsort(x)).astype(np.float32),
    lambda x: np.fft.fft(x).real / x.shape[0],
    lambda x: np.fft.fft(x).imag / x.shape[0],
    lambda x: TT(x),
    lambda x: TT2(x),
    lambda x: TT2(TT2(x)),
    lambda x: TT(TT(x)),
    lambda x: TT(TT(TT(TT(x)))),
    lambda x: x * 0.5,
    lambda x: np.flip(x),
    lambda x: np.concatenate((x[-1:], x[:-1])),
    lambda x: np.concatenate((x[1:], x[:1])),
    lambda x: np.concatenate((x[-2:], x[:-2])),
    lambda x: np.concatenate((x[2:], x[:2])),
    lambda x: np.concatenate((x[-4:], x[:-4])),
    lambda x: np.concatenate((x[4:], x[:4])),
    lambda x: np.concatenate((x[-8:], x[:-8])),
    lambda x: np.concatenate((x[8:], x[:8])),
    lambda x: np.concatenate((x[-16:], x[:-16])),
    lambda x: np.concatenate((x[16:], x[:16])),
    lambda x: np.concatenate((x[-32:], x[:-32])),
    lambda x: np.concatenate((x[32:], x[:32])),
    lambda x: np.mean(x, dtype=np.float64) + x * 0,
    lambda x: (np.log(np.std(x) + 1e-12)).astype(np.float64) + x * 0,
    lambda x: x * 0.1,
    lambda x: (x - np.mean(x)) / (np.std(x) + 1e-12),
    lambda x: x / np.mean(x ** 2) ** 0.5,
    lambda x: x * 10,
    lambda x: x ** 3,
    lambda x: np.sin(x * np.pi),
    lambda x: x - 1,
    lambda x: x * 0.9,
    lambda x: x * 1.1,
    lambda x: np.sign(x) * np.abs(x) ** (1/3),
    lambda x: np.tanh(x),
    lambda x: x * -0.01,
    lambda x: x * 0.01,
    lambda x: np.fft.irfft(np.abs(np.fft.rfft(x)) ** 2),
    lambda x: np.cumsum(x) / (np.arange(x.size, dtype=np.float64)+1.0),
    lambda x: np.cumsum(x),
    lambda x: np.cumprod(x / np.sqrt(np.mean(x ** 2) + 1e-12)),
    lambda x: np.abs(x),
    lambda x: np.maximum(x, 0),
    lambda x: x * len(x),
    lambda x: x / len(x),
    lambda x: np.take(TT(np.take(x, np.argsort(x))), np.argsort(np.argsort(x))),
    lambda x: np.abs(x)**(1/3) * np.sign(x),
]

funcs_2 = [
    lambda x, y: x + y,
    lambda x, y: x - y,
    lambda x, y: x * y,
    lambda x, y: x / (y ** 2 + 1e-12),
    lambda x, y: x / (np.abs(y) + 1e-12),
    lambda x, y: (x - y) ** 2,
    lambda x, y: (x + y) / 2,
    lambda x, y: x * 0.1 + y * 0.9,
    lambda x, y: x * 0.9 + y * 0.1,
    lambda x, y: x * 1.5 + y * -0.5,
    lambda x, y: x * 1.1 + y * -0.1,
    lambda x, y: x * -0.1 + y * 1.1,
    lambda x, y: x * -0.5 + y * 1.5,
    lambda x, y: np.sqrt(x ** 2 + y ** 2),
    lambda x, y: np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(y) / x.shape[0]).real,
    lambda x, y: np.fft.irfft(np.fft.rfft(x) * np.conj(np.fft.rfft(y)) / x.shape[0]).real,
    lambda x, y: np.fft.irfft(np.fft.rfft(y) * np.conj(np.fft.rfft(x)) / x.shape[0]).real,
    lambda x, y: np.fft.irfft(np.fft.rfft(x) ** 2 / (np.fft.rfft(y) + 1e-12)).real,
    lambda x, y: np.fft.irfft(np.fft.rfft(x).real + np.fft.rfft(y).imag * 1j).real,
    lambda x, y: np.fft.irfft(np.fft.rfft(y).real + np.fft.rfft(x).imag * 1j).real,
    lambda x, y: np.take(x, np.argsort(y)),
    lambda x, y: np.take(TT(np.take(x, np.argsort(y))), np.argsort(np.argsort(y))),
    lambda x, y: np.maximum(x, y),
    lambda x, y: np.minimum(x, y),
    lambda x, y: np.sin(x * np.pi * y),
    lambda x, y: attn_poly3_fast(x, y, y),
    lambda x, y: gg(attn_poly3_fast(x**3, y**3, y**3)),
    lambda x, y: attn_poly5_fast(x, y, y),
    lambda x, y: ggg(attn_poly5_fast(x**5, y**5, y**5)),
    lambda x, y: attn_poly11_fast(x, y, y),
    lambda x, y: gggg(attn_poly11_fast(x**11, y**11, y**11)),
]

funcs_3 = [
    lambda x, y, z: np.sqrt((x - y) ** 2 + (y - z) ** 2 + (z - x) ** 2),
    lambda x, y, z: (x + y + z) / 3,
    lambda x, y, z: np.sign(x * y * z) * np.abs(x * y * z) ** (1/3),
    lambda x, y, z: x + y - z,
    lambda x, y, z: x + (y - z) * 0.5,
    lambda x, y, z: np.sqrt(x ** 2 + y ** 2 + z ** 2),
    lambda x, y, z: np.fft.irfft(np.fft.rfft(x) * np.conj(np.fft.rfft(y) / (np.fft.rfft(z + 1e-12)))).real,
    lambda x, y, z: np.fft.irfft(np.fft.rfft(x) * np.conj(np.fft.rfft(np.tanh(y)) / (np.fft.rfft(np.tanh(z))))).real,
    lambda x, y, z: np.fft.irfft(np.fft.rfft(x) * np.conj(np.fft.rfft(np.tanh(y)+1) / (np.fft.rfft(np.tanh(z)+1)))).real,
    lambda x, y, z: np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(y) / (np.fft.rfft(z + 1e-12))).real,
    lambda x, y, z: np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(np.tanh(y)) / (np.fft.rfft(np.tanh(z)))).real,
    lambda x, y, z: np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(np.tanh(y)+1) / (np.fft.rfft(np.tanh(z)+1))).real,
    lambda x, y, z: attn_poly3_fast(x**3, y**3, z),
    lambda x, y, z: gg(attn_poly3_fast(x, y**3, z)),
    lambda x, y, z: attn_poly3_fast(x, y, z),
    lambda x, y, z: gg(attn_poly3_fast(x**3, y**3, z**3)),
    lambda x, y, z: attn_poly5_fast(x, y, z),
    lambda x, y, z: ggg(attn_poly5_fast(x**5, y**5, z**5)),
    lambda x, y, z: attn_poly5_fast(x**5, y**5, z),
    lambda x, y, z: ggg(attn_poly5_fast(x, y**5, z)),
    lambda x, y, z: attn_poly11_fast(x, y, z),
    lambda x, y, z: gggg(attn_poly11_fast(x**11, y**11, z**11)),
    lambda x, y, z: attn_poly11_fast(x**11, y**11, z),
    lambda x, y, z: gggg(attn_poly11_fast(x, y**11, z)),
    lambda x, y, z: np.fft.irfft(attn_poly3_fast(np.fft.rfft(x), np.fft.rfft(y), np.fft.rfft(z))),
    lambda x, y, z: np.fft.irfft(attn_poly5_fast(np.fft.rfft(x), np.fft.rfft(y), np.fft.rfft(z))),
    lambda x, y, z: np.fft.irfft(attn_poly11_fast(np.fft.rfft(x), np.fft.rfft(y), np.fft.rfft(z))),
    lambda x, y, z: np.take(np.take(x, np.argsort(y)), np.argsort(np.argsort(z))),
    lambda x, y, z: np.take(TT(np.take(x, np.argsort(y))), np.argsort(np.argsort(z))),
    lambda x, y, z: np.take(TT(TT(np.take(x, np.argsort(y)))), np.argsort(np.argsort(z))),
    lambda x, y, z: np.take(TT2(np.take(x, np.argsort(y))), np.argsort(np.argsort(z))),
    lambda x, y, z: np.take(TT2(TT2(np.take(x, np.argsort(y)))), np.argsort(np.argsort(z))),
    
]

i0t = funcs_1
i1t = funcs_2
i2t = funcs_3
len_i0 = len(i0t)
len_i1 = len(i1t)
len_i2 = len(i2t)

# =========================================================
# Function sampling distribution T (踏襲)
# =========================================================

def build_T_distribution_1d(
    i0t, i1t, i2t,
    L=1024,
    repeats=400,
    warmup=5,
    power=0.625,
    seed=0,
    eps=1e-12,
):
    rng = np.random.default_rng(seed)

    def _bench_unary(f):
        x = rng.normal(0, 1, size=L).astype(np.float32)
        for _ in range(warmup):
            y = f(x)
            x = np.asarray(y, dtype=np.float32).ravel()
            if x.size != L:
                x = np.resize(x, L).astype(np.float32, copy=False)
        t0 = time.perf_counter()
        for _ in range(repeats):
            x = rng.normal(0, 1, size=L).astype(np.float32)
            y = f(x)
            y = np.asarray(y, dtype=np.float32).ravel()
            if y.size != L:
                y = np.resize(y, L)
        t1 = time.perf_counter()
        return (t1 - t0) / repeats

    def _bench_binary(f):
        x = rng.normal(0, 1, size=L).astype(np.float32)
        y = rng.normal(0, 1, size=L).astype(np.float32)
        for _ in range(warmup):
            z = f(x, y)
            z = np.asarray(z, dtype=np.float32).ravel()
            if z.size != L:
                z = np.resize(z, L)
        t0 = time.perf_counter()
        for _ in range(repeats):
            x = rng.normal(0, 1, size=L).astype(np.float32)
            y = rng.normal(0, 1, size=L).astype(np.float32)
            z = f(x, y)
            z = np.asarray(z, dtype=np.float32).ravel()
            if z.size != L:
                z = np.resize(z, L)
        t1 = time.perf_counter()
        return (t1 - t0) / repeats

    def _bench_ternary(f):
        x = rng.normal(0, 1, size=L).astype(np.float32)
        y = rng.normal(0, 1, size=L).astype(np.float32)
        z = rng.normal(0, 1, size=L).astype(np.float32)
        for _ in range(warmup):
            w = f(x, y, z)
            w = np.asarray(w, dtype=np.float32).ravel()
            if w.size != L:
                w = np.resize(w, L)
        t0 = time.perf_counter()
        for _ in range(repeats):
            x = rng.normal(0, 1, size=L).astype(np.float32)
            y = rng.normal(0, 1, size=L).astype(np.float32)
            z = rng.normal(0, 1, size=L).astype(np.float32)
            w = f(x, y, z)
            w = np.asarray(w, dtype=np.float32).ravel()
            if w.size != L:
                w = np.resize(w, L)
        t1 = time.perf_counter()
        return (t1 - t0) / repeats

    times = []
    for f in i0t:
        try:   dt = _bench_unary(f)
        except Exception: dt = 1e9
        times.append(dt)
    for f in i1t:
        try:   dt = _bench_binary(f)
        except Exception: dt = 1e9
        times.append(dt)
    for f in i2t:
        try:   dt = _bench_ternary(f)
        except Exception: dt = 1e9
        times.append(dt)

    times = np.asarray(times, dtype=np.float64)
    w = 1.0 / (np.maximum(times, eps) ** power)
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones_like(w)
    T = (w / w.sum()).astype(np.float64)
    return T, times

T_dist, times = build_T_distribution_1d(i0t, i1t, i2t)
print("T_dist built:", T_dist.shape)

print(T_dist)
print(np.sum(T_dist[:len_i0]))
print(np.sum(T_dist[len_i0:len_i0+len_i1]))
print(np.sum(T_dist[len_i0+len_i1:]))