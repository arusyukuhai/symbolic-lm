# ============================================================
#  FASTER VERSION (drop-in style) + READOUT(W,b) + score_logits (RESTORED)
#  - PE cache
#  - precompute_structs_numba_fast: no idxs/struct_to_nodes_pair
#  - u64 hash key (much lower collision risk)
#  - topo_sort REMOVED (sid creation order is already topo)
#  - restrict BFS moved OUTSIDE per-image loop: build_needed_sids_once()
#  - batch_exec -> FEATURES directly (POP,last_k)  [NOT class logits]
#  - NEW: per-individual linear readout (W,b): logits10 = W@feat + b
#  - RESTORED: score_logits (your weighted top-k style), NOT cross-entropy
#  NOTE:
#    1) Keep your i0__/i1_/i2_ dictionaries EXACTLY as-is.
#       (If any dict uses PE, it will now call cached PE(a).)
#    2) This file is long; search "PASTE YOUR FUNCTION DICTS HERE".
# ============================================================

import numpy as np
import cv2
import time
import gc
import tqdm
import torchvision
from numba import njit
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.simplefilter('ignore')

# -----------------------------
# Dataset
# -----------------------------
ds = torchvision.datasets.STL10
trainset = ds(root="data", split="train", download=True)
testset  = ds(root="data", split="test",  download=True)

# -----------------------------
# PE cache (IMPORTANT)
# -----------------------------
_pe_cache = {}
def PE(a):
    h, w = a.shape[:2]
    key = (h, w)
    v = _pe_cache.get(key)
    if v is None:
        xs = np.linspace(0.0, 1.0, w, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, h, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)
        _pe_cache[key] = (X, Y)
        v = _pe_cache[key]
    return v

TT  = lambda a: np.concatenate((np.ones(1), (a[:-2] + a[1:-1]*2 + a[2:]) * 0.50, np.ones(1)))
TT2 = lambda a: np.concatenate((np.ones(1), ((a[:-2] - a[1:-1]) ** 2 + (a[1:-1] - a[2:]) ** 2) * 0.5, np.ones(1)))

# =========================
# Stable-ish polynomial "attention"
# =========================
def _attn_poly_fast(k, v, q, deg: int):
    k = np.nan_to_num(k, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float64).ravel()
    v = np.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float64).ravel()
    q = np.nan_to_num(q, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float64).ravel()

    scale = np.max(np.abs(k)) + 1e-12
    kn = k / scale
    qn = q / scale

    size = deg + 1
    max_pow = 2 * deg

    with np.errstate(over='ignore', invalid='ignore'):
        P = kn[:, None] ** np.arange(max_pow + 1, dtype=np.float64)

    if not np.all(np.isfinite(P)):
        return np.zeros_like(q, dtype=np.float64)

    A = np.empty((size, size), dtype=np.float64)
    for i in range(size):
        A[i] = P[:, i:i+size].sum(axis=0)

    b = (v[:, None] * P[:, :size]).sum(axis=0)

    ridge = 1e-8 * np.trace(A) / size if size > 0 else 1e-10
    A = A + np.eye(size, dtype=np.float64) * max(ridge, 1e-10)

    try:
        coef = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        try:
            coef = np.linalg.lstsq(A, b, rcond=1e-15)[0]
        except np.linalg.LinAlgError:
            return np.zeros_like(q, dtype=np.float64)

    Q = qn[:, None] ** np.arange(size, dtype=np.float64)
    return (Q @ coef).astype(np.float64)

def attn_poly3_fast(k, v, q):  return _attn_poly_fast(np.nan_to_num(k), np.nan_to_num(v), np.nan_to_num(q), deg=3)
def attn_poly5_fast(k, v, q):  return _attn_poly_fast(np.nan_to_num(k), np.nan_to_num(v), np.nan_to_num(q), deg=5)
def attn_poly11_fast(k, v, q): return _attn_poly_fast(np.nan_to_num(k), np.nan_to_num(v), np.nan_to_num(q), deg=11)

irr   = lambda x: np.abs(x)**(1/3) * np.sign(x)
irrr  = lambda x: np.abs(x)**(1/5) * np.sign(x)
irrrr = lambda x: np.abs(x)**(1/11) * np.sign(x)

# ============================================================
# PASTE YOUR FUNCTION DICTS HERE (i0__, i1_, i2_) EXACTLY AS-IS
# ============================================================
# --- BEGIN: PASTE YOUR ORIGINAL i0__/i1_/i2_ HERE ---
# (Use exactly what you pasted previously; do not change contents.)

i0__ = {
    "A": lambda a: a ** 2,
    "B": lambda a: np.abs(a),
    "C": lambda a: np.sin(a * np.pi),
    "D": lambda a: np.concatenate((a[-1:], a[:-1]), axis=0),
    "E": lambda a: np.concatenate((a[1:], a[:1]), axis=0),
    "F": lambda a: np.concatenate((a[:, -1:], a[:, :-1]), axis=1),
    "G": lambda a: np.concatenate((a[:, 1:], a[:, :1]), axis=1),
    "H": lambda a: np.fliplr(a),
    "I": lambda a: np.flipud(a),
    "J": lambda a: a * 2,
    "K": lambda a: a * 10,
    "L": lambda a: a * 0.9,
    "M": lambda a: a * 0.1,
    "N": lambda a: a + 1,
    "O": lambda a: -a,
    "P": lambda a: cv2.GaussianBlur(a, (3, 3), 0),
    "Q": lambda a: cv2.GaussianBlur(a, (7, 7), 0),
    "R": lambda a: a * 0 + np.std(a),
    "S": lambda a: a * 0 + np.mean(a),
    "T": lambda a: np.exp(- (PE(a)[0]**2 + PE(a)[1]**2) / (np.var(a) + 0.01)),
    "U": lambda a: a * 0 + np.max(a),
    "V": lambda a: a * 0 + np.min(a),
    "W": lambda a: (a - np.mean(a, axis=0, keepdims=True)) / (np.std(a, axis=0, keepdims=True) + 0.01),
    "X": lambda a: (a - np.mean(a)) / (np.std(a) + 0.01),
    "Y": lambda a: a * 0 + np.mean(a, axis=0, keepdims=True),
    "Z": lambda a: a * 0 + np.std(a, axis=0, keepdims=True),
    "AA": lambda a: np.concatenate((a[::2], a[1::2]), axis=0),
    "AB": lambda a: np.concatenate((a[:, ::2], a[:, 1::2]), axis=-1),
    "AC": lambda a: a * (np.tanh(a) + 1),
    "AD": lambda a: np.tanh(a),
    "AE": lambda a: np.fft.fft2(a).real / a.shape[0],
    "AF": lambda a: np.fft.fft2(a).imag / a.shape[0],
    "AG": lambda a: np.sort(a.flatten()).reshape(a.shape),
    "AH": lambda a: a[np.argsort(np.mean(a, axis=-1))],
    "AI": lambda a: a[np.argsort(np.std(a, axis=-1))],
    "AJ": lambda a: a[:, np.argsort(np.mean(a, axis=0))],
    "AK": lambda a: a[:, np.argsort(np.std(a, axis=0))],
    "AL": lambda a: cv2.resize(a, (a.shape[1]*2, a.shape[0]*2))[a.shape[0]//2:a.shape[0]*2-a.shape[0]+a.shape[0]//2, a.shape[1]//2:a.shape[1]*2-a.shape[1]+a.shape[1]//2],
    "AM": lambda a: np.flip(a),
    "AN": lambda a: cv2.warpAffine(a, cv2.getRotationMatrix2D((a.shape[1]/2, a.shape[0]/2), 90, 1), (a.shape[1], a.shape[0]), borderMode=cv2.BORDER_REPLICATE),
    "AO": lambda a: np.abs(a) ** 0.5,
    "AP": lambda a: np.concatenate((a[-4:], a[:-4]), axis=0),
    "AQ": lambda a: np.concatenate((a[4:], a[:4]), axis=0),
    "AR": lambda a: np.concatenate((a[:, -4:], a[:, :-4]), axis=1),
    "AS": lambda a: np.concatenate((a[:, 4:], a[:, :4]), axis=1),
    "AT": lambda a: np.abs(a) ** (1/3) * np.sign(a),
    "AU": lambda a: a / (np.mean(a ** 2) + 0.01),
    "AV": lambda a: a / (np.mean(a ** 2) ** 0.5 + 0.01),
    "AW": lambda a: a - 1,
    "AX": lambda a: (a - np.mean(a, axis=1, keepdims=True)) / (np.std(a, axis=1, keepdims=True) + 0.01),
    "AY": lambda a: a + 0.1,
    "AZ": lambda a: a + 0.5,
    "BA": lambda a: a - 0.5,
    "BB": lambda a: a - 1,
    "BC": lambda a: 1 - a,
    "BD": lambda a: a / 2,
    "BE": lambda a: a - np.mean(a),
    "BF": lambda a: cv2.GaussianBlur(a, (9, 11), 0),
    "BG": lambda a: np.concatenate((a[-8:], a[:-8]), axis=0),
    "BH": lambda a: np.concatenate((a[8:], a[:8]), axis=0),
    "BI": lambda a: np.concatenate((a[:, -8:], a[:, :-8]), axis=1),
    "BJ": lambda a: np.concatenate((a[:, 8:], a[:, :8]), axis=1),
    "BH_": lambda a: a * 0.5,
    "BI_": lambda a: a * -0.5,
    "BJ_": lambda a: a.T,
    "BK": lambda a: np.fliplr(a.T),
    "BL": lambda a: np.flipud(a.T),
    "BM": lambda a: a ** 3,
    "BN": lambda a: a + 2.0,
    "BO": lambda a: a + 5.0,
    "BP": lambda a: a - 2.0,
    "BQ": lambda a: np.tanh(a),
    "BR": lambda a: a * np.log(np.square(a) + 1e-6),
    "BS": lambda a: np.concatenate((a[-2:], a[:-2]), axis=0),
    "BT": lambda a: np.concatenate((a[2:], a[:2]), axis=0),
    "BU": lambda a: np.concatenate((a[:, -2:], a[:, :-2]), axis=1),
    "BV": lambda a: np.concatenate((a[:, 2:], a[:, :2]), axis=1),
    "BW": lambda a: a * -2,
    "BX": lambda a: a * -4,
    "BY": lambda a: a * -0.1,
    "BZ": lambda a: a * -0.5,
    "CA": lambda a: ((np.concatenate((a[-1:], a[:-1]), axis=0) - a) ** 2 + (np.concatenate((a[1:], a[:1]), axis=0) - a) ** 2) ** 0.5,
    "CB": lambda a: ((np.concatenate((a[:, -1:], a[:, :-1]), axis=1) - a) ** 2 + (np.concatenate((a[:, 1:], a[:, :1]), axis=1) - a) ** 2) ** 0.5,
    "CC": lambda a: ((np.concatenate((a[:, -1:], a[:, :-1]), axis=1) - a) ** 2 + (np.concatenate((a[:, 1:], a[:, :1]), axis=1) - a) ** 2 + (np.concatenate((a[-1:], a[:-1]), axis=0) - a) ** 2 + (np.concatenate((a[1:], a[:1]), axis=0) - a) ** 2) ** 0.5,
    "CD": lambda a: (a - cv2.GaussianBlur(a, (9, 9), 0)) / np.sqrt(cv2.GaussianBlur(a**2 + 1e-6, (9, 9), 0)),
    "CF": lambda a: (a - cv2.GaussianBlur(a, (13, 13), 0)) / np.sqrt(cv2.GaussianBlur(a**2 + 1e-6, (13, 13), 0)),
    "CG": lambda a: np.sort(a, axis=-1),
    "CH": lambda a: np.sort(a, axis=-2),
    "CI": lambda a: np.argsort(a, axis=-1) / a.shape[0],
    "CJ": lambda a: np.argsort(a, axis=-2) / a.shape[0],
    "CK": lambda a: np.maximum(a, np.mean(a))
}

i1_ = {
    "A": lambda a, b: a + b,
    "B": lambda a, b: a - b,
    "C": lambda a, b: a * b,
    "D": lambda a, b: a / (b**2 + 0.01),
    "E": lambda a, b: np.maximum(a, b),
    "F": lambda a, b: np.minimum(a, b),
    "G": lambda a, b: np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(b) / a.shape[0] ** 2).real,
    "H": lambda a, b: np.fft.ifft2(np.fft.fft2(np.tanh(a) + 1) * np.fft.fft2(np.tanh(b)) / a.shape[0] ** 2).real,
    "I": lambda a, b: (a.T @ (np.tanh(b[:a.shape[0], :a.shape[0]]) + 1)).T / a.shape[0],
    "J": lambda a, b: np.take(a.flatten(), np.asarray(np.floor(np.tanh(b.flatten()) * (b.flatten().shape[0] - 1)), dtype=np.int32)).reshape(a.shape),
    "K": lambda a, b: np.take(a.flatten(), np.argsort(b.flatten())).reshape(a.shape),
    "L": lambda a, b: cv2.filter2D(a, -1, cv2.resize(b, (3, 3)) / 3**2),
    "M": lambda a, b: cv2.filter2D(a, -1, cv2.resize(b, (5, 5)) / 5**2),
    "N": lambda a, b: np.fft.ifft(np.fft.fft(a.flatten()) * np.fft.fft(np.tanh(b).flatten()) / a.shape[0] ** 2).real.reshape(a.shape),
    "O": lambda a, b: np.sin(a * b * np.pi),
    "P": lambda a, b: np.mean(np.sin((np.sin((np.repeat(np.stack(PE(a), axis=-1), a.shape[0]//2, axis=-1) @ a + np.mean(a, axis=0)[None, None]) / np.sqrt(a.shape[0]) * np.pi) @ b.T + np.mean(b, axis=1)[None, None]) / np.sqrt(a.shape[0]) * np.pi), axis=-1),
    "Q": lambda a, b: np.concatenate((a[::2], b[1::2]), axis=0),
    "R": lambda a, b: np.take(np.mean(a, axis=1), np.asarray(np.floor(np.tanh(b.flatten()) * (b.shape[0] - 1)), dtype=np.int32)).reshape(a.shape),
    "T": lambda a, b: np.exp(- ((PE(a)[0]**2 - np.mean(a)) / (np.var(a) + 0.01) + (PE(a)[1]**2 - np.mean(b)) / (np.var(b) + 0.01))),
    "U": lambda a, b: a[np.argsort(np.mean(b, axis=-1))],
    "V": lambda a, b: a[:, np.argsort(np.mean(b, axis=0))],
    "W": lambda a, b: (a.T @ (b[:a.shape[0], :a.shape[0]])).T / a.shape[0],
    "X": lambda a, b: (a - b) ** 2,
    "Y": lambda a, b: cv2.filter2D(a, -1, cv2.resize(b, (11, 11)) / 11**2),
    "Z": lambda a, b: cv2.filter2D(a, -1, cv2.resize(b, (50, 50)) / 50**2),
    "AA": lambda a, b: cv2.warpAffine(a, cv2.getRotationMatrix2D((a.shape[1]/2, a.shape[0]/2), np.mean(b) * 360, np.std(b)), (a.shape[1], a.shape[0]), borderMode=cv2.BORDER_REPLICATE),
    "AB": lambda a, b: cv2.warpPerspective(a, cv2.resize(b, (3, 3)), (a.shape[1], a.shape[0]), borderMode=cv2.BORDER_REPLICATE),
    "AC": lambda a, b: np.concatenate((a[:a.shape[0]//2], b[-b.shape[0]//2:]), axis=0),
    "AD": lambda a, b: np.concatenate((a[:, :a.shape[1]//2], b[:, -b.shape[1]//2:]), axis=1),
    "AE": lambda a, b: np.fft.ifft2(a + b*1j).real * a.shape[0]**2,
    "AF": lambda a, b: a[np.asarray(np.floor((b.shape[0] - 1) * (np.tanh(np.mean(b, axis=-1)) + 1) * 0.5), dtype=np.int32)],
    "AG": lambda a, b: a[:, np.asarray(np.floor((b.shape[1] - 1) * (np.tanh(np.mean(b, axis=0)) + 1) * 0.5), dtype=np.int32)],
    "AH": lambda a, b: (a + b) / 2,
    "AI": lambda a, b: (a ** 2 + b ** 2) ** 0.5,
    "AJ": lambda a, b: np.concatenate((a[:, ::2], b[:, 1::2]), axis=-1),
    "AK": lambda a, b: a * (np.tanh(b) + 1),
    "AL": lambda a, b: b * (np.tanh(a) + 1),
    "AM": lambda a, b: a*2 - b,
    "AN": lambda a, b: a*0.75 + b*0.50,
    "AO": lambda a, b: a*0.333 + b*0.666,
    "AP": lambda a, b: a*0.50 + b*0.75,
    "AQ": lambda a, b: np.abs(a * b * b) ** 1/3 * np.sign(a * b * b),
    "AR": lambda a, b: np.abs(a * a * b) ** 1/3 * np.sign(a * a * b),
    "AS": lambda a, b: np.sin(a * b),
    "AT": lambda a, b: np.sin(a * np.mean(b) * np.pi),
    "AU": lambda a, b: cv2.filter2D(a, -1, (np.tanh(cv2.resize(b, (3, 3))))),
    "AV": lambda a, b: np.take(TT(np.take(a.flatten(), np.argsort(b.flatten()))), np.argsort(np.argsort(b.flatten()))).reshape(a.shape),
    "AW": lambda a, b: np.take(TT(np.take(b.flatten(), np.argsort(a.flatten()))), np.argsort(np.argsort(a.flatten()))).reshape(a.shape),
    "AX": lambda a, b: np.take(TT(TT(np.take(b.flatten(), np.argsort(a.flatten())))), np.argsort(np.argsort(a.flatten()))).reshape(a.shape),
    "AY": lambda a, b: np.take(TT2(np.take(b.flatten(), np.argsort(a.flatten()))), np.argsort(np.argsort(a.flatten()))).reshape(a.shape),
    "AZ": lambda a, b: np.take(TT(TT(TT(TT(TT(TT(TT(TT(np.take(b.flatten(), np.argsort(a.flatten())))))))))), np.argsort(np.argsort(a.flatten()))).reshape(a.shape),
    "BA": lambda a, b: np.take(TT(TT(TT(TT(np.take(b.flatten(), np.argsort(a.flatten())))))), np.argsort(np.argsort(a.flatten()))).reshape(a.shape),
    "BB": lambda a, b: np.take(TT2(TT2(np.take(b.flatten(), np.argsort(a.flatten())))), np.argsort(np.argsort(a.flatten()))).reshape(a.shape),
    "BC": lambda a, b: cv2.filter2D(a, -1, np.tanh(cv2.resize(b, (3, 3))) / 3**2),
    "BD": lambda a, b: cv2.filter2D(a, -1, np.tanh(cv2.resize(b, (5, 5))) / 5**2),
    "BE": lambda a, b: cv2.filter2D(a, -1, np.tanh(cv2.resize(b, (7, 7))) / 7**2),
    "BF": lambda a, b: cv2.filter2D(a, -1, (np.tanh(cv2.resize(b, (3, 3))) + 1) / 3**2),
    "BG": lambda a, b: cv2.filter2D(a, -1, (np.tanh(cv2.resize(b, (5, 5))) + 1) / 5**2),
    "BH": lambda a, b: cv2.filter2D(a, -1, (np.tanh(cv2.resize(b, (7, 7))) + 1) / 7**2),
    "BI": lambda a, b: np.fft.ifft2(np.fft.fft2(a) ** 2 / (np.fft.fft2(b) + 1e-8)).real,
    "BK": lambda a, b: np.fft.ifft2(np.fft.fft2(b) ** 2 / (np.fft.fft2(np.tanh(a)) + 1e-8)).real,
    "BL": lambda a, b: cv2.filter2D(a, -1, cv2.resize(b, (15, 15)) / 15**2),
    "BM": lambda a, b: a - b*2,
    "BN": lambda a, b: a*3 - b*2,
    "BO": lambda a, b: a*4 - b*3,
    "BP": lambda a, b: np.argmax(a, axis=0)[np.array(np.floor(np.tanh(b) * 15.5 + 16), dtype=np.int32)] / len(a),
    "BQ": lambda a, b: np.argmax(a, axis=-1)[np.array(np.floor(np.tanh(b) * 15.5 + 16), dtype=np.int32)] / len(a),
    "BR": lambda a, b: np.argmin(a, axis=0)[np.array(np.floor(np.tanh(b) * 15.5 + 16), dtype=np.int32)] / len(a),
    "BS": lambda a, b: np.argmin(a, axis=-1)[np.array(np.floor(np.tanh(b) * 15.5 + 16), dtype=np.int32)] / len(a),
}

i2_ = {
    "AA": lambda a, b, c: a + b + c,
    "AC": lambda a, b, c: np.sign(a * b * c) * np.abs(a * b * c) ** 1/3,
    "AD": lambda a, b, c: np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(b) / (np.fft.fft2(c) + 1e-8)).real,
    "AF": lambda a, b, c: np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(np.tanh(b)) / (np.fft.fft2(np.tanh(c)) + 1e-8)).real,
    "AH": lambda a, b, c: np.maximum(np.maximum(a, b), c),
    "AI": lambda a, b, c: np.minimum(np.minimum(a, b), c),
    "AJ": lambda a, b, c: ((a - b) ** 2 + (b - c) ** 2 + (c - a) ** 2) ** 1/2,
    "AK": lambda a, b, c: (a ** 2 + b ** 2 + c ** 2) ** 1/2,
    "AN": lambda a, b, c: np.take(np.fft.ifft(np.fft.fft(np.take(b.flatten(), np.argsort(a.flatten()))) * np.fft.fft(np.take(c.flatten(), np.argsort(a.flatten()))) / a.shape[0]**2).real, np.argsort(np.argsort(a.flatten()))).reshape(a.shape),
    "AQ": lambda a, b, c: (a + b + c) / 3,
    "AR": lambda a, b, c: a * (1 - np.tanh(b)) + c * (1 + np.tanh(b)),
    "AU": lambda a, b, c: a @ b.T @ c / a.shape[0]**4,
    "AV": lambda a, b, c: a @ np.tanh(b.T @ c) / a.shape[0]**4,
    "AW": lambda a, b, c: attn_poly5_fast(a.flatten(), b.flatten(), c.flatten()).reshape(a.shape),
    "AX": lambda a, b, c: irrr(attn_poly5_fast(a.flatten()**5, b.flatten()**5, c.flatten()**5).reshape(a.shape)),
    "AY": lambda a, b, c: attn_poly5_fast(a.flatten()**5, b.flatten()**5, c.flatten()).reshape(a.shape),
    "AZ": lambda a, b, c: attn_poly11_fast(a.flatten(), b.flatten(), c.flatten()).reshape(a.shape),
    "BA": lambda a, b, c: irrrr(attn_poly11_fast(a.flatten()**11, b.flatten()**11, c.flatten()**11).reshape(a.shape)),
    "BB": lambda a, b, c: attn_poly11_fast(a.flatten()**11, b.flatten()**11, c.flatten()).reshape(a.shape),
    "BC": lambda a, b, c: attn_poly3_fast(a.flatten(), b.flatten(), c.flatten()).reshape(a.shape),
    "BD": lambda a, b, c: irr(attn_poly3_fast(a.flatten()**3, b.flatten()**3, c.flatten()**3).reshape(a.shape)),
    "BE": lambda a, b, c: attn_poly3_fast(a.flatten()**3, b.flatten()**3, c.flatten()).reshape(a.shape),
    "BF": lambda a, b, c: np.fft.ifft2(attn_poly3_fast(np.fft.fft2(a).flatten(), np.fft.fft2(b).flatten(), np.fft.fft2(c).flatten()).reshape(a.shape)).real,
    "BG": lambda a, b, c: np.fft.ifft2(attn_poly5_fast(np.fft.fft2(a).flatten(), np.fft.fft2(b).flatten(), np.fft.fft2(c).flatten()).reshape(a.shape)).real,
    "BH": lambda a, b, c: np.fft.ifft2(attn_poly11_fast(np.fft.fft2(a).flatten(), np.fft.fft2(b).flatten(), np.fft.fft2(c).flatten()).reshape(a.shape)).real,
    "BI": lambda a, b, c: np.take(np.take(a.flatten(), np.argsort(b.flatten())), np.argsort(np.argsort(c.flatten()))).reshape(a.shape),
    "BJ": lambda a, b, c: np.take(TT(np.take(a.flatten(), np.argsort(b.flatten()))), np.argsort(np.argsort(c.flatten()))).reshape(a.shape),
    "BK": lambda a, b, c: np.take(TT2(np.take(a.flatten(), np.argsort(b.flatten()))), np.argsort(np.argsort(c.flatten()))).reshape(a.shape),
    "BL": lambda a, b, c: np.take(TT(TT(np.take(a.flatten(), np.argsort(b.flatten())))), np.argsort(np.argsort(c.flatten()))).reshape(a.shape),
    "BM": lambda a, b, c: np.take(TT(TT(TT(TT(np.take(a.flatten(), np.argsort(b.flatten())))))), np.argsort(np.argsort(c.flatten()))).reshape(a.shape),
    "BN": lambda a, b, c: np.take(TT(TT(TT(TT(TT(TT(TT(TT(np.take(a.flatten(), np.argsort(b.flatten())))))))))), np.argsort(np.argsort(c.flatten()))).reshape(a.shape),
}
# --- end original function dicts ---

# --- END: PASTE YOUR ORIGINAL i0__/i1_/i2_ HERE ---

# Build tables
i0t = list(i0__.values())
i1t = list(i1_.values())
i2t = list(i2_.values())

len_i0 = len(i0t)
len_i1 = len(i1t)
len_i2 = len(i2t)

# -----------------------------
# Function speed sampling -> T
# -----------------------------
def build_T_distribution(i0t, i1t, i2t, rounds=1):
    G = []
    t2 = np.random.normal(0, 1, (96, 96)).astype(np.float32)
    for f in i0t:
        g0 = time.perf_counter()
        for _ in range(rounds):
            f(t2)
        G.append((time.perf_counter() - g0) / rounds)
        t2 = np.random.normal(0, 1, (96, 96)).astype(np.float32)

    for f in i1t:
        g0 = time.perf_counter()
        for _ in range(rounds):
            f(t2, t2)
        G.append((time.perf_counter() - g0) / rounds)
        t2 = np.random.normal(0, 1, (96, 96)).astype(np.float32)

    for f in i2t:
        g0 = time.perf_counter()
        for _ in range(rounds):
            f(t2, t2, t2)
        G.append((time.perf_counter() - g0) / rounds)
        t2 = np.random.normal(0, 1, (96, 96)).astype(np.float32)

    G = np.asarray(G, dtype=np.float64)
    T = 1 / np.maximum(G, 1e-12)
    T = T / T.sum()
    return T, G

T, G_time = build_T_distribution(i0t, i1t, i2t, rounds=80)
print("T built. nonzero ratio:", float((T > 0).mean()))


MODELLEN = 65536
POP = 1
LAST_K = 10   # feature dim
NUM_FUNCS = len_i0 + len_i1 + len_i2

pop_g1 = np.empty((POP, MODELLEN, 3), dtype=np.int32)
pop_g2 = np.empty((POP, MODELLEN), dtype=np.int32)
pop_g3 = np.zeros((POP, MODELLEN), dtype=np.float32)  # fixed zero in your current use

# per-individual linear classifier (NEW; keeps your score_logits objective viable)
pop_w = (np.random.randn(POP, 10, LAST_K) * 0.1).astype(np.float32)
pop_b = np.zeros((POP, 10), dtype=np.float32)

def init_population():
    for p in range(POP):
        for node in range(MODELLEN):
            if node <= 2:
                pop_g1[p, node, 0] = 0
                pop_g1[p, node, 1] = 1
                pop_g1[p, node, 2] = 2
            else:
                pop_g1[p, node, 0] = np.random.randint(0, node)
                pop_g1[p, node, 1] = np.random.randint(0, node)
                pop_g1[p, node, 2] = np.random.randint(0, node)
        pop_g2[p] = np.random.choice(NUM_FUNCS, size=(MODELLEN,), p=T).astype(np.int32)

def init_population():
    for p in range(POP):
        for node in range(MODELLEN):
            if node <= 2:
                pop_g1[p, node, 0] = 0
                pop_g1[p, node, 1] = 1
                pop_g1[p, node, 2] = 2
            else:
                pop_g1[p, node, 0] = np.random.randint(0, node)
                pop_g1[p, node, 1] = np.random.randint(0, node)
                pop_g1[p, node, 2] = np.random.randint(0, node)
        pop_g2[p] = np.random.choice(NUM_FUNCS, size=(MODELLEN,), p=T).astype(np.int32)

init_population()

def active_node(GENE1, GENE2, N_INPUTS, LAST_K):
    I = np.arange(len(GENE1) - LAST_K, len(GENE1))
    Nodes = set(I)
    ANodes = list(I)
    while len(ANodes) > 0:
        BNodes = set()
        for N in ANodes:
            if(GENE1[int(N)][0] >= N_INPUTS):
                BNodes.add(GENE1[int(N)][0])
            if(GENE1[int(N)][1] >= N_INPUTS and GENE2[int(N)] >= len(i0__)):
                BNodes.add(GENE1[int(N)][1])
            if(GENE1[int(N)][2] >= N_INPUTS and GENE2[int(N)] >= len(i0__) + len(i1_)):
                BNodes.add(GENE1[int(N)][2])
        Nodes |= BNodes
        ANodes = list(BNodes)
    return list(np.floor(np.sort(list(Nodes))))

def gene_to_surrogate_input(GENE1, GENE2, N_INPUTS, LAST_K):
    active = active_node(GENE1, GENE2, N_INPUTS, LAST_K)
    surrogate_input = []
    surrogate_input_2 = {}
    for i in range(N_INPUTS):
        onehot = np.zeros(len(i0__)+len(i1_)+len(i2_)+LAST_K+N_INPUTS)
        onehot[len(i0__)+len(i1_)+len(i2_)+LAST_K+i] = 1
        surrogate_input.append(onehot)
        surrogate_input_2[i] = onehot
    for N in active:
        onehot = np.zeros(len(i0__)+len(i1_)+len(i2_)+LAST_K+N_INPUTS)
        onehot += surrogate_input_2[GENE1[int(N)][0]]
        if(GENE2[int(N)] >= len(i0__)):
            onehot += surrogate_input_2[GENE1[int(N)][1]]
        if(GENE2[int(N)] >= len(i0__) + len(i1_)):
            onehot += surrogate_input_2[GENE1[int(N)][2]]
        if(N > len(GENE1) - LAST_K):
            onehot[int(len(i0__)+len(i1_)+len(i2_)+N - len(GENE1) + LAST_K)] += 1
        onehot[GENE2[int(N)]] += 1
        surrogate_input.append(onehot)
        surrogate_input_2[N] = onehot
    return np.stack(surrogate_input)

print(len(active_node(pop_g1[0], pop_g2[0], 3, 10)))
print(gene_to_surrogate_input(pop_g1[0], pop_g2[0], 3, 10).shape)
