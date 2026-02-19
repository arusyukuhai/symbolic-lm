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

# Stable-ish polynomial "attention"
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
    lambda x: np.fft.ifft(np.abs(np.fft.fft(x + 0j)) ** 2).real,
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
    lambda x, y: np.fft.ifft(np.fft.fft(x) * np.fft.fft(y) / x.shape[0]).real,
    lambda x, y: np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(y)) / x.shape[0]).real,
    lambda x, y: np.fft.ifft(np.fft.fft(y) * np.conj(np.fft.fft(x)) / x.shape[0]).real,
    lambda x, y: np.fft.ifft(np.fft.fft(x) ** 2 / (np.fft.fft(y) + 1e-12)).real,
    lambda x, y: np.fft.ifft(np.fft.fft(x).real + np.fft.fft(y).imag * 1j).real,
    lambda x, y: np.fft.ifft(np.fft.fft(y).real + np.fft.fft(x).imag * 1j).real,
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
    lambda x, y, z: np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(y) / (np.fft.fft(z + 1e-12)))).real,
    lambda x, y, z: np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(np.tanh(y)) / (np.fft.fft(np.tanh(z))))).real,
    lambda x, y, z: np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(np.tanh(y)+1) / (np.fft.fft(np.tanh(z)+1)))).real,
    lambda x, y, z: np.fft.ifft(np.fft.fft(x) * np.fft.fft(y) / (np.fft.fft(z + 1e-12))).real,
    lambda x, y, z: np.fft.ifft(np.fft.fft(x) * np.fft.fft(np.tanh(y)) / (np.fft.fft(np.tanh(z)))).real,
    lambda x, y, z: np.fft.ifft(np.fft.fft(x) * np.fft.fft(np.tanh(y)+1) / (np.fft.fft(np.tanh(z)+1))).real,
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
    L=64,
    repeats=80,
    warmup=5,
    power=0.75,
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

# =========================================================
# Structured execution engine (核心は踏襲)
# 変更点: logitsを (N,last_k,L) で返す
# =========================================================

QUANT_BITS = 12
QUANT_SCALE = (1 << QUANT_BITS) - 1

@njit(cache=True)
def compute_used_nodes_numba(G1, G2, MODELLEN, last_k, len_i0, len_i1, num_inputs):
    N = G1.shape[0]
    used = np.zeros((N, MODELLEN), dtype=np.int8)
    stack = np.empty(MODELLEN, dtype=np.int32)

    for ind in range(N):
        top = 0
        start = MODELLEN - last_k
        if start < num_inputs:
            start = num_inputs
        for s in range(start, MODELLEN):
            stack[top] = s
            top += 1

        while top > 0:
            top -= 1
            n = stack[top]
            if used[ind, n] == 1:
                continue
            used[ind, n] = 1
            if n < num_inputs:
                continue

            func_id = int(G2[ind, n])
            if func_id < len_i0:
                a = int(abs(G1[ind, n, 0]))
                if 0 <= a < MODELLEN and used[ind, a] == 0:
                    stack[top] = a; top += 1
            elif func_id < (len_i0 + len_i1):
                a = int(abs(G1[ind, n, 0]))
                b = int(abs(G1[ind, n, 1]))
                if 0 <= a < MODELLEN and used[ind, a] == 0:
                    stack[top] = a; top += 1
                if 0 <= b < MODELLEN and used[ind, b] == 0:
                    stack[top] = b; top += 1
            else:
                a = int(abs(G1[ind, n, 0]))
                b = int(abs(G1[ind, n, 1]))
                c = int(abs(G1[ind, n, 2]))
                if 0 <= a < MODELLEN and used[ind, a] == 0:
                    stack[top] = a; top += 1
                if 0 <= b < MODELLEN and used[ind, b] == 0:
                    stack[top] = b; top += 1
                if 0 <= c < MODELLEN and used[ind, c] == 0:
                    stack[top] = c; top += 1

        for j in range(num_inputs):
            used[ind, j] = 1

    return used

@njit(cache=True)
def _mix64(z):
    # splitmix64-ish
    z = (z ^ (z >> 30)) * np.int64(0xbf58476d1ce4e5b9)
    z = (z ^ (z >> 27)) * np.int64(0x94d049bb133111eb)
    z = z ^ (z >> 31)
    return z

@njit(cache=True)
def _hash_insert_get_sid(keys, vals, key, next_sid, mask):
    h = (key ^ (key >> 33)) & mask
    while True:
        k = keys[h]
        if k == -1:
            keys[h] = key
            vals[h] = next_sid
            return next_sid, 1, next_sid + 1
        elif k == key:
            return vals[h], 0, next_sid
        else:
            h = (h + 1) & mask

@njit(cache=True)
def precompute_structs_numba(G1, G2, G3, len_i0, len_i1, len_i2, num_inputs, last_k=10):
    N = G1.shape[0]
    MODELLEN = G1.shape[1]
    used = compute_used_nodes_numba(G1, G2, MODELLEN, last_k, len_i0, len_i1, num_inputs)

    total_used = 0
    for ind in range(N):
        for node in range(MODELLEN):
            if used[ind, node] == 1:
                total_used += 1

    size = 1
    while size < total_used * 4:
        size <<= 1
    mask = size - 1

    key_table_keys = np.empty(size, dtype=np.int64)
    key_table_vals = np.full(size, -1, dtype=np.int32)
    for i in range(size):
        key_table_keys[i] = -1

    max_S = total_used + num_inputs + 32
    struct_type  = np.empty(max_S, dtype=np.int32)
    struct_func  = np.empty(max_S, dtype=np.int32)
    struct_ch1   = np.empty(max_S, dtype=np.int32)
    struct_ch2   = np.empty(max_S, dtype=np.int32)
    struct_ch3   = np.empty(max_S, dtype=np.int32)
    struct_alpha = np.empty(max_S, dtype=np.float32)

    next_sid = num_inputs
    for sid in range(num_inputs):
        struct_type[sid]  = 0
        struct_func[sid]  = -1
        struct_ch1[sid]   = -1
        struct_ch2[sid]   = -1
        struct_ch3[sid]   = -1
        struct_alpha[sid] = 0.0

    node_structs = np.full((N, MODELLEN), -1, dtype=np.int32)

    pairs_ind  = np.empty(total_used + 1, dtype=np.int32)
    pairs_node = np.empty(total_used + 1, dtype=np.int32)
    pair_pos = 0

    for node in range(MODELLEN):
        for ind in range(N):
            if used[ind, node] == 0:
                continue

            if node < num_inputs:
                node_structs[ind, node] = node
                pairs_ind[pair_pos] = ind
                pairs_node[pair_pos] = node
                pair_pos += 1
                continue

            func_id = int(G2[ind, node])

            a_quant = int(np.floor(G3[ind, node] * QUANT_SCALE + 0.5))
            if a_quant < 0: a_quant = 0
            if a_quant > QUANT_SCALE: a_quant = QUANT_SCALE

            # ---- collision-resistant 64-bit key (replaces bit-packing) ----
            if func_id < len_i0:
                a = int(abs(G1[ind, node, 0]))
                child_sid = node_structs[ind, a]
                key = _mix64(np.int64(1))
                key ^= _mix64(np.int64(func_id + 0x10000))
                key ^= _mix64(np.int64(child_sid + 0x20000))
                key ^= _mix64(np.int64(a_quant + 0x30000))

            elif func_id < (len_i0 + len_i1):
                bi = func_id - len_i0
                a = int(abs(G1[ind, node, 0])); b = int(abs(G1[ind, node, 1]))
                child_a = node_structs[ind, a]
                child_b = node_structs[ind, b]
                key = _mix64(np.int64(2))
                key ^= _mix64(np.int64(bi + 0x40000))
                key ^= _mix64(np.int64(child_a + 0x50000))
                key ^= _mix64(np.int64(child_b + 0x60000))
                key ^= _mix64(np.int64(a_quant + 0x70000))

            else:
                ci = func_id - (len_i0 + len_i1)
                a = int(abs(G1[ind, node, 0])); b = int(abs(G1[ind, node, 1])); c = int(abs(G1[ind, node, 2]))
                child_a = node_structs[ind, a]
                child_b = node_structs[ind, b]
                child_c = node_structs[ind, c]
                key = _mix64(np.int64(3))
                key ^= _mix64(np.int64(ci + 0x80000))
                key ^= _mix64(np.int64(child_a + 0x90000))
                key ^= _mix64(np.int64(child_b + 0xA0000))
                key ^= _mix64(np.int64(child_c + 0xB0000))
                key ^= _mix64(np.int64(a_quant + 0xC0000))

            sid, is_new, next_sid = _hash_insert_get_sid(key_table_keys, key_table_vals, key, next_sid, mask)

            if is_new == 1:
                if func_id < len_i0:
                    struct_type[sid] = 1
                    struct_func[sid] = func_id
                    struct_ch1[sid] = child_sid
                    struct_ch2[sid] = -1
                    struct_ch3[sid] = -1
                elif func_id < (len_i0 + len_i1):
                    struct_type[sid] = 2
                    struct_func[sid] = func_id - len_i0
                    struct_ch1[sid] = child_a
                    struct_ch2[sid] = child_b
                    struct_ch3[sid] = -1
                else:
                    struct_type[sid] = 3
                    struct_func[sid] = func_id - (len_i0 + len_i1)
                    struct_ch1[sid] = child_a
                    struct_ch2[sid] = child_b
                    struct_ch3[sid] = child_c

                struct_alpha[sid] = float(a_quant) / float(QUANT_SCALE)

            node_structs[ind, node] = sid
            pairs_ind[pair_pos] = ind
            pairs_node[pair_pos] = node
            pair_pos += 1

    S = next_sid
    struct_type  = struct_type[:S].copy()
    struct_func  = struct_func[:S].copy()
    struct_ch1   = struct_ch1[:S].copy()
    struct_ch2   = struct_ch2[:S].copy()
    struct_ch3   = struct_ch3[:S].copy()
    struct_alpha = struct_alpha[:S].copy()

    # map struct -> nodes (kept)
    counts = np.zeros(S, dtype=np.int32)
    for p in range(pair_pos):
        sid = node_structs[pairs_ind[p], pairs_node[p]]
        counts[sid] += 1
    idxs = np.empty(S + 1, dtype=np.int32)
    idxs[0] = 0
    for s in range(S):
        idxs[s+1] = idxs[s] + counts[s]
    M = idxs[-1]
    struct_to_nodes_pair = np.empty((M, 2), dtype=np.int32)
    write_pos = idxs[:-1].copy()
    for p in range(pair_pos):
        ind = pairs_ind[p]
        node = pairs_node[p]
        sid = node_structs[ind, node]
        pos = write_pos[sid]
        struct_to_nodes_pair[pos, 0] = ind
        struct_to_nodes_pair[pos, 1] = node
        write_pos[sid] += 1

    return node_structs, struct_type, struct_func, struct_ch1, struct_ch2, struct_ch3, struct_alpha, idxs, struct_to_nodes_pair

@njit(cache=True)
def topo_sort_structs_numba_from_arrays(struct_type, struct_ch1, struct_ch2, struct_ch3):
    S = struct_type.shape[0]
    parent_count = np.zeros(S, dtype=np.int32)

    for sid in range(S):
        t = struct_type[sid]
        if t == 1:
            c = struct_ch1[sid]
            if c >= 0: parent_count[c] += 1
        elif t == 2:
            c1 = struct_ch1[sid]; c2 = struct_ch2[sid]
            if c1 >= 0: parent_count[c1] += 1
            if c2 >= 0: parent_count[c2] += 1
        elif t == 3:
            c1 = struct_ch1[sid]; c2 = struct_ch2[sid]; c3 = struct_ch3[sid]
            if c1 >= 0: parent_count[c1] += 1
            if c2 >= 0: parent_count[c2] += 1
            if c3 >= 0: parent_count[c3] += 1

    tot = 0
    offsets = np.empty(S, dtype=np.int32)
    for i in range(S):
        offsets[i] = tot
        tot += parent_count[i]
    parent_buf = np.empty(tot, dtype=np.int32)

    for i in range(S):
        parent_count[i] = 0

    for sid in range(S):
        t = struct_type[sid]
        if t == 1:
            c = struct_ch1[sid]
            if c >= 0:
                idx = offsets[c] + parent_count[c]
                parent_buf[idx] = sid
                parent_count[c] += 1
        elif t == 2:
            c1 = struct_ch1[sid]; c2 = struct_ch2[sid]
            if c1 >= 0:
                idx = offsets[c1] + parent_count[c1]
                parent_buf[idx] = sid
                parent_count[c1] += 1
            if c2 >= 0:
                idx = offsets[c2] + parent_count[c2]
                parent_buf[idx] = sid
                parent_count[c2] += 1
        elif t == 3:
            c1 = struct_ch1[sid]; c2 = struct_ch2[sid]; c3 = struct_ch3[sid]
            if c1 >= 0:
                idx = offsets[c1] + parent_count[c1]
                parent_buf[idx] = sid
                parent_count[c1] += 1
            if c2 >= 0:
                idx = offsets[c2] + parent_count[c2]
                parent_buf[idx] = sid
                parent_count[c2] += 1
            if c3 >= 0:
                idx = offsets[c3] + parent_count[c3]
                parent_buf[idx] = sid
                parent_count[c3] += 1

    indeg = np.zeros(S, dtype=np.int32)
    for sid in range(S):
        t = struct_type[sid]
        if t == 1:
            indeg[sid] += (struct_ch1[sid] >= 0)
        elif t == 2:
            indeg[sid] += (struct_ch1[sid] >= 0)
            indeg[sid] += (struct_ch2[sid] >= 0)
        elif t == 3:
            indeg[sid] += (struct_ch1[sid] >= 0)
            indeg[sid] += (struct_ch2[sid] >= 0)
            indeg[sid] += (struct_ch3[sid] >= 0)

    q = np.empty(S, dtype=np.int32)
    ql = 0; qr = 0
    for s in range(S):
        if indeg[s] == 0:
            q[qr] = s; qr += 1

    out = np.empty(S, dtype=np.int32)
    out_len = 0
    while ql < qr:
        s = q[ql]; ql += 1
        out[out_len] = s; out_len += 1
        start = offsets[s]
        end = offsets[s] + parent_count[s]
        for pidx in range(start, end):
            p = parent_buf[pidx]
            indeg[p] -= 1
            if indeg[p] == 0:
                q[qr] = p; qr += 1

    if out_len < S:
        res = np.empty(out_len, dtype=np.int32)
        for i in range(out_len):
            res[i] = out[i]
        return res
    return out

def _as_vec32(y, L):
    if isinstance(y, np.ndarray):
        if y.dtype != np.float32:
            y = y.astype(np.float32, copy=False)
        y = y.ravel()
        if y.size != L:
            y = np.resize(y, L).astype(np.float32, copy=False)
        return y
    y = np.asarray(y, dtype=np.float32).ravel()
    if y.size != L:
        y = np.resize(y, L).astype(np.float32, copy=False)
    return y

def batch_exec_structured_logits_1d_dense(
    x_inputs: np.ndarray,   # (num_inputs, L)
    node_structs,
    struct_type,
    struct_func,
    struct_ch1,
    struct_ch2,
    struct_ch3,
    struct_alpha,
    topo,
    last_k,
    restrict=True,
):
    x_inputs = np.asarray(x_inputs, dtype=np.float32)
    if x_inputs.ndim != 2:
        raise ValueError("x_inputs must be (num_inputs, L)")
    num_inputs, L = x_inputs.shape

    N = node_structs.shape[0]
    MODELLEN = node_structs.shape[1]
    S = struct_type.shape[0]

    # needed mask
    if restrict:
        needed = np.zeros(S, dtype=np.bool_)
        start0 = max(num_inputs, MODELLEN - last_k)
        q = np.empty(N * last_k + 1024, dtype=np.int32)
        qlen = 0

        for ind in range(N):
            for ln in range(start0, MODELLEN):
                sid = int(node_structs[ind, ln])
                if sid >= 0 and not needed[sid]:
                    needed[sid] = True
                    if qlen >= q.size:
                        q = np.resize(q, q.size * 2)
                    q[qlen] = sid
                    qlen += 1

        qi = 0
        while qi < qlen:
            s = int(q[qi]); qi += 1
            c1 = int(struct_ch1[s]); c2 = int(struct_ch2[s]); c3 = int(struct_ch3[s])
            if c1 >= 0 and not needed[c1]:
                needed[c1] = True
                if qlen >= q.size: q = np.resize(q, q.size * 2)
                q[qlen] = c1; qlen += 1
            if c2 >= 0 and not needed[c2]:
                needed[c2] = True
                if qlen >= q.size: q = np.resize(q, q.size * 2)
                q[qlen] = c2; qlen += 1
            if c3 >= 0 and not needed[c3]:
                needed[c3] = True
                if qlen >= q.size: q = np.resize(q, q.size * 2)
                q[qlen] = c3; qlen += 1
    else:
        needed = None

    outputs = [None] * S
    for i in range(min(num_inputs, S)):
        outputs[i] = x_inputs[i]

    _i0t = i0t; _i1t = i1t; _i2t = i2t
    _stype = struct_type
    _sf = struct_func
    _c1a = struct_ch1; _c2a = struct_ch2; _c3a = struct_ch3
    _alpha = struct_alpha

    for sid in topo:
        sid = int(sid)
        if sid < 0 or sid >= S:
            continue
        if needed is not None and not needed[sid]:
            continue
        t = int(_stype[sid])
        if t == 0:
            continue

        a = float(_alpha[sid])

        if t == 1:
            fid = int(_sf[sid])
            c1 = int(_c1a[sid])
            x = outputs[c1]
            base = _i0t[fid](x)
            base = _as_vec32(base, L)
            outputs[sid] = base * (1.0 - a) + x * a

        elif t == 2:
            fid = int(_sf[sid])
            c1 = int(_c1a[sid]); c2 = int(_c2a[sid])
            x = outputs[c1]; y = outputs[c2]
            base = _i1t[fid](x, y)
            base = _as_vec32(base, L)
            outputs[sid] = base * (1.0 - a) + x * a

        elif t == 3:
            fid = int(_sf[sid])
            c1 = int(_c1a[sid]); c2 = int(_c2a[sid]); c3 = int(_c3a[sid])
            x = outputs[c1]; y = outputs[c2]; z = outputs[c3]
            base = _i2t[fid](x, y, z)
            base = _as_vec32(base, L)
            outputs[sid] = base * (1.0 - a) + x * a
        else:
            raise RuntimeError(f"unknown struct type: {t}")

    # Collect last_k node sids -> logits (N,last_k,L)
    last_nodes = np.arange(max(0, MODELLEN - last_k), MODELLEN, dtype=np.int32)
    last_sids = node_structs[:, last_nodes]  # (N,last_k)

    logits = np.zeros((N, last_k, L), dtype=np.float32)
    for i in range(N):
        for j in range(last_k):
            s = int(last_sids[i, j])
            if s >= 0:
                arr = outputs[s]
                if arr is not None:
                    logits[i, j, :] = _as_vec32(arr, L)
    return logits

# =========================================================
# D3PM forward corruption (uniform replacement)
# q(x_t | x_{t-1}) : keep w.p. (1-beta_t), else uniform random token
# =========================================================

def make_beta_schedule(T=64, beta_start=1e-4, beta_end=0.03):
    # simple linear
    betas = np.linspace(beta_start, beta_end, T, dtype=np.float64)
    betas = np.clip(betas, 1e-8, 0.5)
    return betas

def corrupt_step(x_prev, beta_t, V, rng):
    # x_prev: (L,) int
    L = x_prev.shape[0]
    keep = rng.random(L) >= beta_t
    x = x_prev.copy()
    # replace where not keep
    idx = np.where(~keep)[0]
    if idx.size > 0:
        x[idx] = rng.integers(0, V, size=idx.size, dtype=np.int32)
    return x

def sample_x_from_x0_level(x0, alpha_bar, V, rng):
    # sample x_level from x0 with keep prob alpha_bar, else uniform
    L = x0.shape[0]
    keep = rng.random(L) < alpha_bar
    x = x0.copy()
    idx = np.where(~keep)[0]
    if idx.size > 0:
        x[idx] = rng.integers(0, V, size=idx.size, dtype=np.int32)
    return x

# =========================================================
# Inputs builder: (num_inputs, L)
#  - token one-hot for x_t: channels [0..V-1]
#  - extra channels: t_norm, noise_level, pos, ones
# =========================================================

def build_inputs_from_xt(x_t, t, T, V, L, pos_cache=None):
    num_inputs = V + 4
    x_inputs = np.zeros((num_inputs, L), dtype=np.float32)

    # token one-hot
    x_inputs[x_t, np.arange(L)] = 1.0

    t_norm = np.float32(t / T)
    x_inputs[V + 0, :] = t_norm

    # a "noise_level" proxy (monotone): t/T
    x_inputs[V + 1, :] = t_norm

    if pos_cache is None or pos_cache.shape[0] != L:
        pos = (np.arange(L, dtype=np.float32) / max(1, (L - 1))).astype(np.float32)
    else:
        pos = pos_cache
    x_inputs[V + 2, :] = pos

    x_inputs[V + 3, :] = 1.0
    return x_inputs

# =========================================================
# Loss: cross-entropy for target x_{t-1}
# logits: (POP,V,L), targets: (L,)
# =========================================================

def ce_loss_pop(logits_pop, targets):
    # logits_pop: (POP,V,L)
    POP, V, L = logits_pop.shape
    # logsumexp over V
    m = np.max(logits_pop, axis=1)  # (POP,L)
    ex = np.exp(logits_pop - m[:, None, :])
    lse = m + np.log(np.sum(ex, axis=1) + 1e-12)  # (POP,L)
    # gather target logits
    pop_idx = np.arange(POP)[:, None]
    pos_idx = np.arange(L)[None, :]
    tgt_logits = logits_pop[pop_idx, targets[None, :], pos_idx]  # (POP,L)
    nll = (lse - tgt_logits).mean(axis=1)  # (POP,)
    return nll

# =========================================================
# GA loop for GP-D3PM
# =========================================================

import tqdm
def run_gp_d3pm(
    MODELLEN=4096,
    POP=128,
    iters=2000,
    L=128,          # block size
    batch=32,       # samples per generation eval
    T=64,           # diffusion steps
    beta_start=1e-4,
    beta_end=0.03,
    seed=0,
    restrict=True,
):
    rng = np.random.default_rng(seed)

    # dataset
    text = load_tiny_shakespeare("input.txt")
    train_txt, val_txt, test_txt = split_train_val_test(text)
    chars, stoi, itos = build_vocab(train_txt)
    V = len(chars)
    print("Vocab size:", V)

    train_ids = encode(train_txt, stoi)

    # diffusion schedule
    betas = make_beta_schedule(T=T, beta_start=beta_start, beta_end=beta_end)  # length T
    alphas = 1.0 - betas
    alpha_bar = np.ones(T + 1, dtype=np.float64)
    for t in range(1, T + 1):
        alpha_bar[t] = alpha_bar[t-1] * alphas[t-1]

    last_k = V  # output channels
    num_inputs = V + 4

    if MODELLEN < num_inputs + last_k + 8:
        raise ValueError(f"MODELLEN too small: need at least {num_inputs + last_k + 8}")

    pos_cache = (np.arange(L, dtype=np.float32) / max(1, (L - 1))).astype(np.float32)

    # init genes
    GENES1, GENES2, GENES3 = [], [], []
    for _ in range(POP):
        # DAG refs biased to smaller indices
        G1 = np.abs((1 - rng.uniform(0, 1, (MODELLEN, 3))**1.25) * (np.arange(MODELLEN)[:, None]))
        G2 = rng.choice(len_i0 + len_i1 + len_i2, size=(MODELLEN,), p=T_dist)
        G3 = rng.uniform(0, 1, size=(MODELLEN,)).astype(np.float32)

        GENES1.append(G1.astype(np.int64))
        GENES2.append(G2.astype(np.int64))
        GENES3.append(G3.astype(np.float32))

    elites = []
    history = []

    def sample_batch_x0():
        # random contiguous chunks
        max_start = train_ids.shape[0] - (L + 1)
        starts = rng.integers(0, max_start, size=batch, dtype=np.int32)
        x0 = np.stack([train_ids[s:s+L] for s in starts], axis=0)  # (batch,L)
        return x0

    for step in range(iters):
        # stack population
        G1 = np.stack(GENES1, axis=0)
        G2 = np.stack(GENES2, axis=0)
        G3 = np.zeros_like(np.stack(GENES3, axis=0))  # <- (重要) alphaをちゃんと使う

        node_structs, struct_type, struct_func, struct_ch1, struct_ch2, struct_ch3, struct_alpha, idxs, pairs = \
            precompute_structs_numba(G1, G2, G3, len_i0, len_i1, len_i2, num_inputs=num_inputs, last_k=last_k)
        topo = topo_sort_structs_numba_from_arrays(struct_type, struct_ch1, struct_ch2, struct_ch3)

        losses = np.zeros(POP, dtype=np.float64)

        x0_batch = sample_batch_x0()  # (batch,L)

        for b in tqdm.tqdm(range(batch)):
            x0 = x0_batch[b]

            # sample t in [1..T]
            t = int(rng.integers(1, T + 1))
            # sample x_{t-1} from x0 at level t-1 (closed form)
            x_tm1 = sample_x_from_x0_level(x0, alpha_bar[t-1], V, rng)
            # one-step corrupt to x_t
            x_t = corrupt_step(x_tm1, betas[t-1], V, rng)

            x_inputs = build_inputs_from_xt(x_t, t, T, V, L, pos_cache=pos_cache)
            logits = np.nan_to_num(batch_exec_structured_logits_1d_dense(
                x_inputs,
                node_structs, struct_type, struct_func, struct_ch1, struct_ch2, struct_ch3,
                struct_alpha, topo,
                last_k=last_k,
                restrict=restrict,
            )) / 64  # (POP,V,L)

            losses += ce_loss_pop(logits, x_tm1)

        losses /= batch

        rank = np.argsort(losses)
        best = int(rank[0])
        best_loss = float(losses[best])

        history.append(best_loss)

        # keep elite sometimes
        if (len(elites) == 0) or (best_loss < elites[-1][-1] - 1e-6):
            elites_allow = (deepcopy(GENES1[best]), deepcopy(GENES2[best]), deepcopy(GENES3[best]), best_loss)
            elites.append(elites_allow)

        print(f"step {step} | best CE: {best_loss:.4f} | elites: {len(elites)}")

        # ---- reproduction ----
        new1, new2, new3 = [], [], []

        # keep top few
        keep = min(4, POP)
        for k in range(keep):
            idx = int(rank[k])
            new1.append(deepcopy(GENES1[idx]))
            new2.append(deepcopy(GENES2[idx]))
            new3.append(deepcopy(GENES3[idx]))

        # inject recent elites
        for e in elites[-min(8, len(elites)):]:
            new1.append(deepcopy(e[0])); new2.append(deepcopy(e[1])); new3.append(deepcopy(e[2]))

        # fill rest by crossover+mutation
        while len(new1) < POP:
            p1 = int(rank[rng.integers(0, max(1, int(np.sqrt(POP))))])
            p2 = int(rank[rng.integers(0, max(1, int(np.sqrt(POP))))])

            c1 = deepcopy(GENES1[p1]); c2 = deepcopy(GENES2[p1]); c3 = deepcopy(GENES3[p1])

            a = int(rng.integers(num_inputs, MODELLEN-1))
            b = int(rng.integers(a, MODELLEN))
            c1[a:b] = GENES1[p2][a:b]
            c2[a:b] = GENES2[p2][a:b]
            mix = float(rng.uniform(0, 1))
            c3[a:b] = c3[a:b] * mix + GENES3[p2][a:b] * (1-mix)

            # mutations (ほぼ踏襲)
            if rng.random() < 0.25:
                for _ in range(int(rng.integers(2, 2**int(rng.integers(2, max(3, int(np.log2(MODELLEN)))))))):
                    pos = int(rng.integers(num_inputs, MODELLEN))
                    which = int(rng.integers(0, 3))
                    c1[pos, which] = int(rng.integers(0, pos))  # DAG
            if rng.random() < 0.25:
                for _ in range(int(rng.integers(2, 2**int(rng.integers(2, max(3, int(np.log2(MODELLEN)))))))):
                    pos = int(rng.integers(num_inputs, MODELLEN))
                    c2[pos] = int(rng.choice(len_i0 + len_i1 + len_i2, p=T_dist))
            if rng.random() < 0.30:
                for _ in range(int(rng.integers(1, 6))):
                    pos = int(rng.integers(num_inputs, MODELLEN))
                    c3[pos] = np.float32(np.clip(c3[pos] + rng.normal(0, 0.2), 0.0, 1.0))
            if np.random.rand() < 0.05:
                pos1 = np.random.randint(num_inputs, MODELLEN-2)
                pos2 = np.random.randint(pos1, MODELLEN)
                c2[pos1:pos2] = np.random.choice(len_i0 + len_i1 + len_i2, size=(pos2-pos1,), p=T_dist)
            if np.random.rand() < 0.05:
                pos1 = np.random.randint(num_inputs, MODELLEN-2)
                pos2 = np.random.randint(pos1, MODELLEN)
                c1[pos1:pos2] = np.random.randint(0, pos1, size=(pos2-pos1,3))
            if np.random.rand() < 0.05:
                size = 2**np.random.randint(0, int(np.log2(MODELLEN)-2))
                pos1 = np.random.randint(num_inputs+size, MODELLEN-size-1)
                pos2 = np.random.randint(pos1, MODELLEN-size)
                pos3 = np.random.randint(0, size)
                c1[pos1+pos3:pos2+pos3] = c1[pos1:pos2]
                c2[pos1+pos3:pos2+pos3] = c2[pos1:pos2]
                c3[pos1+pos3:pos2+pos3] = c3[pos1:pos2]
            if np.random.rand() < 0.05:
                size = 2**np.random.randint(0, int(np.log2(MODELLEN)-2))
                pos1 = np.random.randint(num_inputs+size, MODELLEN-size-1)
                pos2 = np.random.randint(pos1, MODELLEN-size)
                pos3 = np.random.randint(0, size)
                c1[pos1+pos3:pos2+pos3] = c1[pos1:pos2]+pos3
                c2[pos1+pos3:pos2+pos3] = c2[pos1:pos2]
                c3[pos1+pos3:pos2+pos3] = c3[pos1:pos2]

            new1.append(c1); new2.append(c2); new3.append(np.clip(c3, 0.0, 1.0).astype(np.float32))

        GENES1, GENES2, GENES3 = new1[:POP], new2[:POP], new3[:POP]

        if step % 10 == 0 and step > 0:
            np.savez("gp_d3pm_state.npz", GENES1=GENES1, GENES2=GENES2, GENES3=GENES3, history=np.array(history, dtype=np.float32))
        gc.collect()

        if step % 10 == 0:
            best_elite = min(elites, key=lambda e: e[-1])

            out = sample_from_gp_d3pm(best_elite, (chars, stoi, itos, betas, T, L), seed=42, prefix="ROMEO:")
            print("---- sample ----")
            print(out)

    # return best elite and helpers
    best_elite = min(elites, key=lambda e: e[-1])
    return best_elite, (chars, stoi, itos, betas, T, L)

# =========================================================
# Sampling: start from uniform noise x_T, then iterate t=T..1
# =========================================================

def sample_from_gp_d3pm(best_elite, meta, steps=None, seed=123, prefix=""):
    G1b, G2b, G3b, best_loss = best_elite
    chars, stoi, itos, betas, T, L = meta
    V = len(chars)
    last_k = V
    num_inputs = V + 4
    rng = np.random.default_rng(seed)

    if steps is None:
        steps = T

    # Build structs for single individual
    G1 = G1b[None, ...].astype(np.int64)
    G2 = G2b[None, ...].astype(np.int64)
    G3 = G3b[None, ...].astype(np.float32)

    node_structs, struct_type, struct_func, struct_ch1, struct_ch2, struct_ch3, struct_alpha, idxs, pairs = \
        precompute_structs_numba(G1, G2, G3, len_i0, len_i1, len_i2, num_inputs=num_inputs, last_k=last_k)
    topo = topo_sort_structs_numba_from_arrays(struct_type, struct_ch1, struct_ch2, struct_ch3)

    pos_cache = (np.arange(L, dtype=np.float32) / max(1, (L - 1))).astype(np.float32)

    # init x_T as uniform
    x = rng.integers(0, V, size=L, dtype=np.int32)

    # optional prefix clamp (inpainting-ish)
    if prefix:
        p_ids = np.array([stoi.get(c, 0) for c in prefix], dtype=np.int32)
        m = min(L, p_ids.size)
        x[:m] = p_ids[:m]

    # reverse steps
    for t in range(min(steps, T), 0, -1):
        x_inputs = build_inputs_from_xt(x, t, T, V, L, pos_cache=pos_cache)
        logits = np.nan_to_num(batch_exec_structured_logits_1d_dense(
            x_inputs,
            node_structs, struct_type, struct_func, struct_ch1, struct_ch2, struct_ch3,
            struct_alpha, topo,
            last_k=last_k,
            restrict=True,
        )[0]) / 64  # (V,L)

        # sample x_{t-1} from softmax(logits)
        m = np.max(logits, axis=0)  # (L,)
        probs = np.exp(logits - m[None, :])
        probs = probs / (np.sum(probs, axis=0, keepdims=True) + 1e-12)  # (V,L)

        # categorical sample per position
        # (naive loop; L=128 なら十分)
        x_new = x.copy()
        for i in range(L):
            x_new[i] = int(rng.choice(V, p=probs[:, i]))
        x = x_new

        # keep prefix fixed if given
        if prefix:
            m2 = min(L, len(prefix))
            x[:m2] = p_ids[:m2]

    return decode(x, itos)

# =========================================================
# Main (tiny sanity)
# =========================================================
if __name__ == "__main__":
    # 小さめで動作確認 → だんだん上げる
    best_elite, meta = run_gp_d3pm(
        MODELLEN=20000,
        POP=512,
        iters=80000,
        L=512,
        batch=32,
        T=64,
        beta_end=0.03,
        seed=0,
        restrict=True,
    )
    print("best CE:", best_elite[-1])
