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
import os
import scipy
from flask import Flask
import threading

torch.set_num_threads(1)

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

torch.distributed.init_process_group("gloo", rank=0, world_size=1)

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
    "CK": lambda a: np.maximum(a, np.mean(a)),
    "CL": lambda a: scipy.special.erf(a),
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
def build_T_distribution(i0t, i1t, i2t, rounds=10):
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
    T = 1 / np.maximum(G, 1e-12) ** 0.5
    T = T / T.sum()
    return T, G

T, G_time = build_T_distribution(i0t, i1t, i2t, rounds=10)
print("T built. nonzero ratio:", float((T > 0).mean()))

# ============================================================
# QUANT
# ============================================================
QUANT_BITS  = 12
QUANT_SCALE = (1 << QUANT_BITS) - 1

# ============================================================
# NUMBA: used-nodes marking
# ============================================================
@njit(cache=True)
def compute_used_nodes_numba(G1, G2, MODELLEN, last_k, len_i0, len_i1):
    N = G1.shape[0]
    used = np.zeros((N, MODELLEN), dtype=np.int8)
    stack = np.empty(MODELLEN, dtype=np.int32)
    for ind in range(N):
        top = 0
        start = MODELLEN - last_k
        if start < 3:
            start = 3
        for s in range(start, MODELLEN):
            stack[top] = s
            top += 1
        while top > 0:
            top -= 1
            n = stack[top]
            if used[ind, n] == 1:
                continue
            used[ind, n] = 1
            if n <= 2:
                continue

            func_id = int(G2[ind, n])

            if func_id < len_i0:
                a = int(G1[ind, n, 0])
                if used[ind, a] == 0:
                    stack[top] = a; top += 1
            elif func_id < (len_i0 + len_i1):
                a = int(G1[ind, n, 0]); b = int(G1[ind, n, 1])
                if used[ind, a] == 0:
                    stack[top] = a; top += 1
                if used[ind, b] == 0:
                    stack[top] = b; top += 1
            else:
                a = int(G1[ind, n, 0]); b = int(G1[ind, n, 1]); c = int(G1[ind, n, 2])
                if used[ind, a] == 0:
                    stack[top] = a; top += 1
                if used[ind, b] == 0:
                    stack[top] = b; top += 1
                if used[ind, c] == 0:
                    stack[top] = c; top += 1

        used[ind, 0] = 1
        used[ind, 1] = 1
        used[ind, 2] = 1
    return used

# ============================================================
# NUMBA: u64 hashing (splitmix64)
# ============================================================
@njit(cache=True)
def splitmix64(x):
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = x
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return z ^ (z >> np.uint64(31))

@njit(cache=True)
def make_key_u64(stype, func_id, c1, c2, c3, aq):
    key = np.uint64(0)
    key ^= splitmix64(np.uint64(stype))
    key ^= splitmix64(np.uint64(func_id) ^ (np.uint64(aq) << np.uint64(32)))
    key ^= splitmix64(np.uint64(c1) ^ (np.uint64(c2) << np.uint64(21)))
    key ^= splitmix64(np.uint64(c3) ^ (np.uint64(aq) << np.uint64(11)))
    return key

@njit(cache=True)
def _hash_insert_get_sid_u64(keys, vals, key, next_sid, table_mask, empty):
    h = np.int64(key) & table_mask
    while True:
        k = keys[h]
        if k == empty:
            keys[h] = key
            vals[h] = next_sid
            return next_sid, 1, next_sid + 1
        elif k == key:
            return vals[h], 0, next_sid
        else:
            h = (h + 1) & table_mask

# ============================================================
# NUMBA: precompute_structs_numba_fast
# ============================================================
@njit(cache=True)
def precompute_structs_numba_fast(G1, G2, G3, len_i0, len_i1, len_i2, last_k=10):
    N = G1.shape[0]
    MODELLEN = G1.shape[1]

    used = compute_used_nodes_numba(G1, G2, MODELLEN, last_k, len_i0, len_i1)

    total_used = 0
    for ind in range(N):
        for node in range(MODELLEN):
            if used[ind, node] == 1:
                total_used += 1

    size = 1
    while size < total_used * 4:
        size <<= 1
    table_mask = size - 1

    empty = np.uint64(0xFFFFFFFFFFFFFFFF)
    key_table_keys = np.empty(size, dtype=np.uint64)
    key_table_vals = np.full(size, -1, dtype=np.int32)
    for i in range(size):
        key_table_keys[i] = empty

    max_S = total_used + 3
    struct_type  = np.empty(max_S, dtype=np.int32)
    struct_func  = np.empty(max_S, dtype=np.int32)
    struct_ch1   = np.empty(max_S, dtype=np.int32)
    struct_ch2   = np.empty(max_S, dtype=np.int32)
    struct_ch3   = np.empty(max_S, dtype=np.int32)
    struct_alpha = np.empty(max_S, dtype=np.float32)

    next_sid = 3
    for sid in range(3):
        struct_type[sid]  = 0
        struct_func[sid]  = -1
        struct_ch1[sid]   = -1
        struct_ch2[sid]   = -1
        struct_ch3[sid]   = -1
        struct_alpha[sid] = 0.0

    node_structs = np.full((N, MODELLEN), -1, dtype=np.int32)

    for node in range(MODELLEN):
        for ind in range(N):
            if used[ind, node] == 0:
                continue

            if node <= 2:
                node_structs[ind, node] = node
                continue

            func_id = int(G2[ind, node])

            aq = int(np.floor(G3[ind, node] * QUANT_SCALE + 0.5))
            if aq < 0: aq = 0
            if aq > QUANT_SCALE: aq = QUANT_SCALE

            if func_id < len_i0:
                stype = 1
                c1 = int(G1[ind, node, 0])
                ch1 = node_structs[ind, c1]
                ch2 = -1
                ch3 = -1
            elif func_id < (len_i0 + len_i1):
                stype = 2
                c1 = int(G1[ind, node, 0]); c2 = int(G1[ind, node, 1])
                ch1 = node_structs[ind, c1]
                ch2 = node_structs[ind, c2]
                ch3 = -1
            else:
                stype = 3
                c1 = int(G1[ind, node, 0]); c2 = int(G1[ind, node, 1]); c3 = int(G1[ind, node, 2])
                ch1 = node_structs[ind, c1]
                ch2 = node_structs[ind, c2]
                ch3 = node_structs[ind, c3]

            key = make_key_u64(stype, func_id, ch1, ch2, ch3, aq)
            sid, is_new, next_sid = _hash_insert_get_sid_u64(
                key_table_keys, key_table_vals, key, next_sid, table_mask, empty
            )

            if is_new == 1:
                struct_type[sid] = stype
                if stype == 1:
                    struct_func[sid] = func_id
                    struct_ch1[sid]  = ch1
                    struct_ch2[sid]  = -1
                    struct_ch3[sid]  = -1
                elif stype == 2:
                    struct_func[sid] = func_id - len_i0
                    struct_ch1[sid]  = ch1
                    struct_ch2[sid]  = ch2
                    struct_ch3[sid]  = -1
                else:
                    struct_func[sid] = func_id - (len_i0 + len_i1)
                    struct_ch1[sid]  = ch1
                    struct_ch2[sid]  = ch2
                    struct_ch3[sid]  = ch3

                struct_alpha[sid] = np.float32(aq / QUANT_SCALE)

            node_structs[ind, node] = sid

    S = next_sid
    return (node_structs,
            struct_type[:S].copy(),
            struct_func[:S].copy(),
            struct_ch1[:S].copy(),
            struct_ch2[:S].copy(),
            struct_ch3[:S].copy(),
            struct_alpha[:S].copy())

# ============================================================
# Build needed_sids ONCE per generation
# ============================================================
def build_needed_sids_once(node_structs, struct_type, ch1, ch2, ch3, last_k=10):
    N, MODELLEN = node_structs.shape
    last_nodes = np.arange(max(0, MODELLEN-last_k), MODELLEN, dtype=np.int32)
    seeds = np.unique(node_structs[:, last_nodes].reshape(-1))
    seeds = seeds[seeds >= 0]

    S = struct_type.shape[0]
    needed = np.zeros(S, dtype=np.bool_)
    needed[:min(3, S)] = True

    stack = [int(s) for s in seeds if s < S]
    for s in stack:
        needed[s] = True

    while stack:
        s = stack.pop()
        t = int(struct_type[s])
        if t == 1:
            c1_ = int(ch1[s])
            if c1_ >= 0 and not needed[c1_]:
                needed[c1_] = True; stack.append(c1_)
        elif t == 2:
            c1_ = int(ch1[s]); c2_ = int(ch2[s])
            if c1_ >= 0 and not needed[c1_]:
                needed[c1_] = True; stack.append(c1_)
            if c2_ >= 0 and not needed[c2_]:
                needed[c2_] = True; stack.append(c2_)
        elif t == 3:
            c1_ = int(ch1[s]); c2_ = int(ch2[s]); c3_ = int(ch3[s])
            if c1_ >= 0 and not needed[c1_]:
                needed[c1_] = True; stack.append(c1_)
            if c2_ >= 0 and not needed[c2_]:
                needed[c2_] = True; stack.append(c2_)
            if c3_ >= 0 and not needed[c3_]:
                needed[c3_] = True; stack.append(c3_)

    needed_sids = np.where(needed)[0].astype(np.int32)
    needed_sids.sort()
    return needed_sids

# ============================================================
# Execute only needed_sids, return FEATURES (POP,last_k)
# ============================================================
def batch_exec_features_fast(input_arr,
                            node_structs,
                            struct_type,
                            struct_func,
                            struct_ch1,
                            struct_ch2,
                            struct_ch3,
                            struct_alpha,
                            needed_sids,
                            last_k=10):
    if input_arr.dtype != np.float32:
        input_arr = input_arr.astype(np.float32, copy=False)

    N, MODELLEN = node_structs.shape
    H, W = input_arr.shape[0], input_arr.shape[1]
    S = struct_type.shape[0]

    outputs = [None] * S
    outputs[0] = input_arr[:, :, 0] if input_arr.ndim == 3 else input_arr
    outputs[1] = input_arr[:, :, 1] if input_arr.ndim == 3 else np.zeros((H, W), np.float32)
    outputs[2] = input_arr[:, :, 2] if input_arr.ndim == 3 else np.zeros((H, W), np.float32)

    for sid in needed_sids:
        sid = int(sid)
        if sid < 3:
            continue
        t = int(struct_type[sid])
        alpha = float(struct_alpha[sid])

        if t == 1:
            fid = int(struct_func[sid])
            c1 = int(struct_ch1[sid])
            a = outputs[c1]
            base = i0t[fid](a)
            outputs[sid] = (1.0 - alpha) * base + alpha * a

        elif t == 2:
            fid = int(struct_func[sid])
            c1 = int(struct_ch1[sid]); c2 = int(struct_ch2[sid])
            a = outputs[c1]; b = outputs[c2]
            base = i1t[fid](a, b)
            outputs[sid] = (1.0 - alpha) * base + alpha * a

        elif t == 3:
            fid = int(struct_func[sid])
            c1 = int(struct_ch1[sid]); c2 = int(struct_ch2[sid]); c3 = int(struct_ch3[sid])
            a = outputs[c1]; b = outputs[c2]; c = outputs[c3]
            base = i2t[fid](a, b, c)
            outputs[sid] = (1.0 - alpha) * base + alpha * a

    last_nodes = np.arange(max(0, MODELLEN-last_k), MODELLEN, dtype=np.int32)
    last_sids = node_structs[:, last_nodes]  # (POP,last_k)

    uniq = np.unique(last_sids[last_sids >= 0])
    sid_mean = np.zeros(S, dtype=np.float32)
    sid_has = np.zeros(S, dtype=np.bool_)
    for usid in uniq:
        usid = int(usid)
        arr = outputs[usid]
        if arr is None:
            continue
        sid_mean[usid] = np.float32(arr.mean())
        sid_has[usid] = True

    feat = np.zeros((N, last_k), dtype=np.float32)
    for i in range(N):
        for j in range(last_k):
            sid = int(last_sids[i, j])
            if sid >= 0 and sid_has[sid]:
                feat[i, j] = sid_mean[sid]
    return feat

# ============================================================
# RESTORED: score_logits (works on class logits (POP,10))
# ============================================================
def score_logits_from_logits10(logits10: np.ndarray, label: int):
    # logits10: (POP,10)
    order = logits10
    top1 = (order[:, 0] == label).astype(np.float32)

    weights = 1 / ((np.arange(10) + 1) ** 3)
    kk = min(9, order.shape[1])
    hit = np.zeros((logits10.shape[0],), dtype=np.float32)
    for k in range(kk):
        hit += (order[:, k] == label).astype(np.float32) * weights[k]
    return top1, hit

# ============================================================
# POPULATION
# ============================================================
MODELLEN = 10000
POP = 20**2
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
        onehot = surrogate_input_2[GENE1[int(N)][0]]
        if(GENE2[int(N)] >= len(i0__)):
            onehot = (surrogate_input_2[GENE1[int(N)][0]] + surrogate_input_2[GENE1[int(N)][1]]) / 2
        if(GENE2[int(N)] >= len(i0__) + len(i1_)):
            onehot = (surrogate_input_2[GENE1[int(N)][0]] + surrogate_input_2[GENE1[int(N)][1]] + surrogate_input_2[GENE1[int(N)][2]]) / 3
        if(N > len(GENE1) - LAST_K):
            onehot[int(len(i0__)+len(i1_)+len(i2_)+N - len(GENE1) + LAST_K)] += 1
        onehot[GENE2[int(N)]] += 1
        surrogate_input.append(onehot)
        surrogate_input_2[N] = onehot
    return np.stack(surrogate_input)

# ============================================================
# TRAIN/TEST SAMPLES
# ============================================================
test_datas = [testset[j] for j in range(512)]

# ============================================================
# BOOKKEEPING (restore style)
# ============================================================
bestacc = np.full(100, 0.0, dtype=np.float64)
elites_g1 = []
elites_g2 = []
elites_w  = []
elites_b  = []

def train_slice_indices(step, total, slices=100):
    a = (step % slices) * total // slices
    b = (step % slices + 1) * total // slices
    return a, b

class Expert(nn.Module):
    def __init__(self, input_m=1):
        super().__init__()
        self.linearA = nn.Linear(192 * input_m, 768)
        self.linearB = nn.Linear(768, 192)
        self.linearC = nn.Linear(768, 192)
        self.linearD = nn.Linear(192 * input_m, 768)
        self.rmsnorm = nn.RMSNorm(192 * input_m)
        
    def forward(self, x):
        X = self.rmsnorm(x)
        X = F.silu(self.linearA(X)) * self.linearD(X)
        gate = F.sigmoid(self.linearB(X))
        return self.linearC(X) * gate + x[..., :192] * (1 - gate)

class SurrogateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_1 = nn.Parameter(torch.randn(192))
        self.embedding_2 = nn.Parameter(torch.randn(192))
        self.embedding_3 = nn.Parameter(torch.randn(192))
        self.i0t_experts = nn.ModuleList([Expert() for _ in range(len(i0t))])
        self.i1t_experts = nn.ModuleList([Expert(2) for _ in range(len(i1t))])
        self.i2t_experts = nn.ModuleList([Expert(3) for _ in range(len(i2t))])
        self.output = nn.Linear(192*10, 768)
        self.output2 = nn.Linear(192*10, 768)
        self.output3 = nn.Linear(768, 1)
        self.rmsnorm = nn.RMSNorm(192*10)

    def forward(self, node_structs,
                            struct_type,
                            struct_func,
                            struct_ch1,
                            struct_ch2,
                            struct_ch3,
                            struct_alpha,
                            needed_sids,
                            last_k = 10):
        # Surrogate Run With Correct DAG
        N, MODELLEN = node_structs.shape
        S = struct_type.shape[0]

        outputs = [None] * S
        outputs[0] = self.embedding_1
        outputs[1] = self.embedding_2
        outputs[2] = self.embedding_3

        for sid in needed_sids:
            sid = int(sid)
            if sid < 3:
                continue
            t = int(struct_type[sid])
            alpha = float(struct_alpha[sid])

            if t == 1:
                fid = int(struct_func[sid])
                c1 = int(struct_ch1[sid])
                a = outputs[c1]
                base = self.i0t_experts[fid](a)
                outputs[sid] = (1.0 - alpha) * base + alpha * a

            elif t == 2:
                fid = int(struct_func[sid])
                c1 = int(struct_ch1[sid]); c2 = int(struct_ch2[sid])
                a = outputs[c1]; b = outputs[c2]
                base = self.i1t_experts[fid](torch.concatenate([a, b], dim=-1))
                outputs[sid] = (1.0 - alpha) * base + alpha * a

            elif t == 3:
                fid = int(struct_func[sid])
                c1 = int(struct_ch1[sid]); c2 = int(struct_ch2[sid]); c3 = int(struct_ch3[sid])
                a = outputs[c1]; b = outputs[c2]; c = outputs[c3]
                base = self.i2t_experts[fid](torch.concatenate([a, b, c], dim=-1))
                outputs[sid] = (1.0 - alpha) * base + alpha * a

        last_nodes = np.arange(max(0, MODELLEN-last_k), MODELLEN, dtype=np.int32)
        last_sids = node_structs[:, last_nodes]  # (POP,last_k)

        uniq = np.unique(last_sids[last_sids >= 0])
        sid_mean = {}
        sid_has = {}
        for usid in uniq:
            usid = int(usid)
            arr = outputs[usid]
            if arr is None:
                continue
            sid_mean[usid] = arr
            sid_has[usid] = True

        feat = []
        for i in range(N):
            G = self.rmsnorm(torch.flatten(torch.concatenate([sid_mean[int(sid)] for sid in last_sids[i] if sid >= 0 and sid_has[int(sid)]])))
            feat.append(torch.mean(F.sigmoid(self.output3(self.output2(G) * F.silu(self.output(G)))), dim=-1))
        feat = torch.stack(feat)
        """feat = np.zeros((N, last_k), dtype=np.float32)
        for i in range(N):
            for j in range(last_k):
                sid = int(last_sids[i, j])
                if sid >= 0 and sid_has[sid]:
                    feat[i, j] = sid_mean[sid]"""
        return feat
        

"""class GaussianDropout(nn.Module):
    def __init__(self, alpha=1.0):
        super(GaussianDropout, self).__init__()
        self.alpha = alpha
        
    def forward(self, x):
        epsilon = torch.randn_like(x) * self.alpha + 1
        return x * epsilon"""

"""def golu(x):
    return x * torch.exp(-torch.exp(torch.clip(-x, None, 12)))

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.xl = nn.Linear(384, 384)
        self.yl = nn.Linear(384, 384)
        self.zl = nn.Linear(384, 384)
        self.wl = nn.Linear(384, 384)
        self.p = GaussianDropout(0.0025)

    def forward(self, x):
        return F.sigmoid(torch.mean(self.wl(golu(self.zl(self.p(torch.mean(self.xl(x) * golu(self.yl(x)), dim=1))))), dim=1))

encoder_layer = nn.TransformerEncoderLayer(d_model=384, nhead=4, activation=golu)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

surrogate_model = nn.Sequential(
    GaussianDropout(0.025),
    nn.Linear(len(i0__)+len(i1_)+len(i2_)+10+3, 384),
    GaussianDropout(0.025),
    transformer_encoder,
    GaussianDropout(0.025),
    CosineSimilarityLoss(),
)
surrogate_model.to("mps")"""
idhk_surrogate = 0

def set_all_singular_values_to_near_one(G, steps: int):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
 
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X
 
def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    update = set_all_singular_values_to_near_one(update, steps=ns_steps)
    update *= (grad.size(-2) / grad.size(-1))**0.5
    return update
 
def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)
 
class Muon(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())
 
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
 
        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                        state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
 
        return loss


import copy

surrogatemodel = SurrogateModel()
surrogatemodel.to("mps")
hidden_weights = [p for p in surrogatemodel.parameters() if p.ndim >= 2 and p.shape[0] != 1 and p.shape[1] != 1]
hidden_gains_biases = [p for p in surrogatemodel.parameters() if p.ndim < 2 or p.shape[0] == 1 or p.shape[1] == 1]
param_groups = [
    dict(params=hidden_weights, use_muon=True,
         lr=0.02, weight_decay=0.01),
    dict(params=hidden_gains_biases, use_muon=False,
         lr=3e-4, betas=(0.9, 0.98), weight_decay=0.01),
]
optimizer = Muon(param_groups)
prevsurrogate_loss = 0
surrogate_history = []  # List of (step, model_state, optimizer_state)
string = "step,min_loss,min_loss_surrogate,avg_loss,prev_loss,max_acc,num_elites,max_esmilated_loss,idhk_surrogate,percent<br/>"

app = Flask(__name__)

@app.route("/")
def index():
    global string
    return "<html><head><meta http-equiv=\"refresh\" content=\"5\"><meta charset=\"UTF-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\"></head><body><pre>" + string + "</pre></body></html>"

t = threading.Thread(target=app.run, args=("localhost", 8000))
t.start()

# ============================================================
# MAIN LOOP
# ============================================================
for step in range(1_000_000):
    losses = np.zeros(POP, dtype=np.float64)  # will accumulate "hit"
    acc1   = np.zeros(POP, dtype=np.float64)

    # ---- PRECOMPUTE STRUCTS ONCE PER GENERATION ----
    (node_structs,
     struct_type,
     struct_func,
     struct_ch1,
     struct_ch2,
     struct_ch3,
     struct_alpha) = precompute_structs_numba_fast(
        pop_g1, pop_g2, pop_g3,
        len_i0, len_i1, len_i2,
        last_k=LAST_K
    )

    needed_sids = build_needed_sids_once(node_structs, struct_type, struct_ch1, struct_ch2, struct_ch3, last_k=LAST_K)

    esmilated_losses = surrogatemodel(node_structs, struct_type, struct_func, struct_ch1, struct_ch2, struct_ch3, struct_alpha,
            needed_sids)
    
    # --- NaN/Low-Std Check and Rollback ---
    es_numpy = esmilated_losses.detach().cpu().numpy()
    if np.isnan(es_numpy).any() or np.isinf(es_numpy).any() or np.std(es_numpy) < 1e-4:
        #print(f"\n[ALERT] Surrogate model instability detected at step {step}!")
        #print(f"NaN/Inf found: {np.isnan(es_numpy).any() or np.isinf(es_numpy).any()}")
        #print(f"Std dev: {np.std(es_numpy)}")
        
        if len(surrogate_history) > 0:
            # Try to roll back to the oldest available state (which should be ~20-30 steps ago)
            target_step, target_model_state, target_opt_state = surrogate_history[0]
            #print(f"Rolling back to step {target_step}...")
            surrogatemodel.load_state_dict(target_model_state)
            optimizer.load_state_dict(target_opt_state)
            
            # Re-run forward pass with restored model
            esmilated_losses = surrogatemodel(node_structs, struct_type, struct_func, struct_ch1, struct_ch2, struct_ch3, struct_alpha,
                    needed_sids)
        else:
            pass
            #print("No history available to roll back. Proceeding with caution.")
    
    # Save history periodically (every 10 steps)
    if step % 5 == 0:
        surrogate_history.append((
            step, 
            {k: v.cpu().clone() for k, v in surrogatemodel.state_dict().items()},
            copy.deepcopy(optimizer.state_dict())
        ))
        # Keep last 3 history snapshots (approx 30 steps of history)
        if len(surrogate_history) > 3:
            surrogate_history.pop(0)
    #print(esmilated_losses.detach().cpu().numpy().shape)
    # ---- TRAIN EVAL ----
    a, b = train_slice_indices(step, len(trainset), slices=100)
    SCALE = 96

    rank = []
    it = list(range(a, b))
    for idx in tqdm.tqdm(it) if step == 0 else it:
        img, label = trainset[idx]
        ds_img = np.array(img)  # uint8 HWC
        if SCALE != ds_img.shape[0]:
            ds_img = cv2.resize(ds_img, (SCALE, SCALE), interpolation=cv2.INTER_AREA)

        inp = (ds_img.astype(np.float32) / 255.0)

        logits10 = batch_exec_features_fast(
            inp,
            node_structs, struct_type, struct_func, struct_ch1, struct_ch2, struct_ch3, struct_alpha,
            needed_sids,
            last_k=LAST_K
        )  # (POP, LAST_K)

        av = np.argsort(logits10[:, :10], axis=-1)
        top1, hit = score_logits_from_logits10(av, label)
        acc1   += top1
        losses += hit

    # normalize like your original
    denom = max(1, (b - a))
    acc1   = acc1 / denom * 100.0
    losses = -(losses / denom) * 100.0  # more negative = better (higher hit)
    losses__ = np.copy(losses)

    g = torch.tensor(-losses / 100, dtype=torch.float32).to("mps")
    surrogate_loss = torch.mean(- (((1 - g) * torch.log(1 - esmilated_losses)) + (g * torch.log(esmilated_losses))))
    surrogate_loss -= torch.mean(- (((1 - g) * torch.log(1 - g)) + (g * torch.log(g))))
    if(step == 0):
        idhk_surrogate = np.log2(surrogate_loss.detach().cpu().numpy())
    idhk_surrogate = np.nan_to_num(np.log2(surrogate_loss.detach().cpu().numpy())) * 0.1 + idhk_surrogate * 0.9

    surrogate_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    percent = scipy.special.erf(step / 300 - 2) * 0.5 + 0.5
    losses = (1 - percent) * losses - percent * esmilated_losses.detach().cpu().numpy() * 100
    
    """optimizer.zero_grad()
    eslosses = np.zeros(POP, dtype=np.float64)
    surrogate_loss = 0
    nan_occurred = False
    prev_state = {k: v.clone() for k, v in surrogate_model.state_dict().items()}
    prev_opt_state = copy.deepcopy(optimizer.state_dict())
    
    # --- Batched Surrogate Inference ---
    # Group indices by input sequence length
    groups = {}
    for i in range(len(losses)):
        s_input = gene_to_surrogate_input(pop_g1[i], pop_g2[i], 3, 10)
        length = str(s_input.shape[0])
        if length not in groups:
            groups[length] = []
        while len(groups[length]) >= 8:
            length = length + "_"
            if length not in groups:
                groups[length] = []
        groups[length].append((i, s_input))

    for length, items in tqdm.tqdm(groups.items()) if step == 0 else groups.items():
        indices = [x[0] for x in items]
        batch_inputs = np.stack([x[1] for x in items])
        surrogate_input = torch.tensor(batch_inputs, dtype=torch.float32).to("mps")
        
        # Forward pass
        # surrogate_model(surrogate_input) -> (Batch, 1) or similar depending on model architecture
        # Looking at original code: esmilated = surrogate_model(surrogate_input)[0] 
        # where surrogate_input was [1, Length, D]
        # Current batch_inputs is [Batch, Length, D]
        esmilated_batch = surrogate_model(surrogate_input) # (Batch,)
        
        # Check for NaN in output
        if torch.isnan(esmilated_batch).any() or torch.isinf(esmilated_batch).any():
            nan_occurred = True
            break
            
        target_losses = torch.tensor([-losses[i] / 100 for i in indices], dtype=torch.float32).to("mps")
        
        # Calculate loss for the batch
        # Original: loss = torch.square(esmilated - (-losses[i]) / 100) / len(losses)
        # We need to sum the squares and divide by total POP to match original gradient scale
        batch_loss = torch.sum(torch.square(esmilated_batch - target_losses)) / len(losses)
        
        # Check for NaN in loss
        if torch.isnan(batch_loss) or torch.isinf(batch_loss):
            nan_occurred = True
            break
            
        batch_loss.backward()
        
        # Update eslosses and losses for each individual in the batch
        for idx_in_batch, i in enumerate(indices):
            val = esmilated_batch[idx_in_batch].item()
            eslosses[i] += val
            losses[i] -= np.nan_to_num(val) * step / 5000 * 100
        
        surrogate_loss += batch_loss.item()
        gc.collect()"""
    rank = np.argsort(losses)  # best first

    # elite tracking (restore your rule shape)
    slot = step % 100
    a_prev = bestacc[slot]
    if bestacc[slot] / (np.min(losses__)) < 0.999999:
        bestacc[slot] = np.min(losses__)
        if step < 100:
            elites_g1.append(pop_g1[rank[0]].copy())
            elites_g2.append(pop_g2[rank[0]].copy())
            elites_w.append(pop_w[rank[0]].copy())
            elites_b.append(pop_b[rank[0]].copy())
        else:
            elites_g1[slot] = pop_g1[rank[0]].copy()
            elites_g2[slot] = pop_g2[rank[0]].copy()
            elites_w[slot]  = pop_w[rank[0]].copy()
            elites_b[slot]  = pop_b[rank[0]].copy()

    esmilated_losses = esmilated_losses.detach().cpu().numpy()
    print(step, ",", -float(np.min(losses__)), ",", -float(np.min(losses)), ",", -float(np.sum(bestacc) / min(step+1, len(bestacc))), ",", -float(a_prev), ",", float(np.max(acc1)), ",", len(elites_g1), ",", float(np.max(esmilated_losses) * 100), ",", idhk_surrogate, ",", percent)
    string += str(step) + "," + str(-float(np.min(losses__))) + "," + str(-float(np.min(losses))) + "," + str(-float(np.sum(bestacc) / min(step+1, len(bestacc)))) + "," + str(-float(a_prev)) + "," + str(float(np.max(acc1))) + "," + str(len(elites_g1)) + "," + str(float(np.max(esmilated_losses) * 100)) + "," + str(idhk_surrogate) + "," + str(percent) + "\n<br/>"

    # ============================================================
    # SELECTION + REPRODUCTION (aligned for g1/g2/w/b)
    # ============================================================
    new_g1 = []
    new_g2 = []
    new_w  = []
    new_b  = []

    KEEP = 20
    for tt in range(KEEP):
        pidx = int(rank[tt])
        new_g1.append(pop_g1[pidx].copy())
        new_g2.append(pop_g2[pidx].copy())
        new_w.append(pop_w[pidx].copy())
        new_b.append(pop_b[pidx].copy())

    mutation_rate = np.random.uniform(0, 1) ** 3

    for g__ in range(19):
        for h__ in range(20):
            g = g__
            h = h__
            pa = int(rank[g])
            pb = int(rank[h])

            child1 = pop_g1[pa].copy()
            child2 = pop_g2[pa].copy()
            childw = pop_w[pa].copy()
            childb = pop_b[pa].copy()

            pos1 = np.random.randint(0, MODELLEN-2)
            pos2 = np.random.randint(pos1+1, MODELLEN-1)

            child1[pos1:pos2] = pop_g1[pb, pos1:pos2]
            child2[pos1:pos2] = pop_g2[pb, pos1:pos2]

            # readout crossover
            k1 = np.random.randint(0, LAST_K-1)
            k2 = np.random.randint(k1+1, LAST_K)
            childw[:, k1:k2] = pop_w[pb, :, k1:k2]
            if np.random.rand() < 0.5:
                childb[:] = pop_b[pb]

            # graph mutations (your spirit, unchanged)
            if np.random.uniform(0, 1) < 0.04 * mutation_rate:
                for _ in range(np.random.randint(1, 7)):
                    pos = np.random.randint(MODELLEN-11, MODELLEN-1)
                    cidx = np.random.randint(0, 3)
                    child1[pos, cidx] = np.random.randint(0, pos)

            if np.random.uniform(0, 1) < 0.03 * mutation_rate:
                for _ in range(np.random.randint(1, 7)):
                    tt2 = 1 - (np.random.uniform(0, 1) ** 2)
                    T2 = (T ** tt2)
                    T2 = T2 / T2.sum()
                    pos = np.random.randint(MODELLEN-11, MODELLEN-1)
                    child2[pos] = np.random.choice(NUM_FUNCS, p=T2)

            if(np.random.uniform(0, 1) < 0.05 * mutation_rate):
                for __ in range(np.random.randint(1, 2**np.random.randint(1, np.floor(np.log2(MODELLEN))))):
                    pos = np.random.randint(1, MODELLEN//3-1) + np.random.randint(1, MODELLEN//3-1) + np.random.randint(1, MODELLEN//3-1)
                    child1[pos][np.random.randint(0, 3)] = np.random.randint(0, pos-1)

            if(np.random.uniform(0, 1) < 0.05 * mutation_rate):
                tt = 1 - (np.random.uniform(0, 1) ** 2)
                T2 = (T ** tt) / np.sum(T ** tt)
                for __ in range(np.random.randint(1, 2**np.random.randint(1, np.floor(np.log2(MODELLEN))))):
                    pos = np.random.randint(1, MODELLEN//3-1) + np.random.randint(1, MODELLEN//3-1) + np.random.randint(1, MODELLEN//3-1)
                    child2[pos] = np.random.choice(NUM_FUNCS, p=T2)

            if(np.random.uniform(0, 1) < 0.0015 * mutation_rate):
                # ensure int32
                tmp = np.abs(np.random.uniform(0, 1, (MODELLEN, 3)) * (np.arange(MODELLEN)[:, None]))
                child1 = tmp.astype(np.int32, copy=False)

            if(np.random.uniform(0, 1) < 0.0015 * mutation_rate):
                tt = 1 - (np.random.uniform(0, 1) ** 2)
                T2 = (T ** tt) / np.sum(T ** tt)
                child2 = np.random.choice(NUM_FUNCS, (MODELLEN,), p=T2).astype(np.int32)

            if(np.random.uniform(0, 1) < 0.003150 * mutation_rate):
                TT3 = 2**np.random.randint(1, np.floor(np.log2(MODELLEN))-1)
                p1 = np.random.randint(0, MODELLEN-2)
                p2 = np.random.randint(p1, MODELLEN)
                lv = np.random.randint(-TT3, TT3)
                child1[p1:p2] = child1[p1:p2] + lv

            if(np.random.uniform(0, 1) < 0.003150 * mutation_rate):
                TT3 = 2**np.random.randint(1, np.floor(np.log2(MODELLEN))-1)
                p1 = np.random.randint(TT3+1, MODELLEN-TT3-1)
                p2 = np.random.randint(p1, MODELLEN-TT3-1)
                p3 = np.random.randint(-TT3, TT3)
                child1[p1-p3:p2-p3] = child1[p1:p2]

            if(np.random.uniform(0, 1) < 0.003150 * mutation_rate):
                TT3 = 2**np.random.randint(1, np.floor(np.log2(MODELLEN))-1)
                p1 = np.random.randint(TT3+1, MODELLEN-TT3-1)
                p2 = np.random.randint(p1, MODELLEN-TT3-1)
                p3 = np.random.randint(-TT3, TT3)
                child2[p1-p3:p2-p3] = child2[p1:p2]

            if(np.random.uniform(0, 1) < 0.003150 * mutation_rate):
                # heavy FFT-mix (keep as you had; ensure int32 at end)
                tmp = np.floor(
                    np.fft.ifft(
                        np.fft.fft(child1.astype(np.float64)+0j, axis=0)
                        * np.fft.fft(pop_g1[rank[h]].astype(np.float64)+0j, axis=0)
                        / (np.fft.fft(pop_g1[np.random.randint(0, POP)].astype(np.float64)+0j, axis=0) + 1e-9),
                        axis=0
                    ).real
                )
                child1 = tmp.astype(np.int32, copy=False)

            if(np.random.uniform(0, 1) < 0.0015 * mutation_rate):
                tmp = np.floor(child1.astype(np.float64) + (pop_g1[rank[h]].astype(np.float64) - pop_g1[np.random.randint(0, POP)].astype(np.float64)) * np.random.uniform(0, 1.5))
                child1 = tmp.astype(np.int32, copy=False)

            if(np.random.uniform(0, 1) < 0.0015 * mutation_rate):
                tmp = np.floor(child1.astype(np.float64) + (pop_g1[rank[h]].astype(np.float64) - pop_g1[np.random.randint(0, POP)].astype(np.float64)))
                child1 = tmp.astype(np.int32, copy=False)

            if(np.random.uniform(0, 1) < 0.00650 * mutation_rate):
                TT3 = 2**np.random.randint(1, np.floor(np.log2(MODELLEN))-1)
                p1 = np.random.randint(TT3+1, MODELLEN-TT3-1)
                p2 = np.random.randint(p1, MODELLEN-TT3-1)
                p3 = np.random.randint(-TT3, TT3)
                child1[p1-p3:p2-p3] = child1[p1:p2]
                child2[p1-p3:p2-p3] = child2[p1:p2]

            if(np.random.uniform(0, 1) < 0.00650 * mutation_rate):
                TT3 = 2**np.random.randint(1, np.floor(np.log2(MODELLEN))-1)
                p1 = np.random.randint(TT3+1, MODELLEN-TT3-1)
                p2 = np.random.randint(p1, MODELLEN-TT3-1)
                p3 = np.random.randint(-TT3, TT3)
                child1[p1-p3:p2-p3] = child1[p1:p2] - p3
                child2[p1-p3:p2-p3] = child2[p1:p2]

            # clamp graph edges valid
            child1 = np.maximum(child1, 0)
            child1 = np.minimum(child1, np.maximum((np.arange(MODELLEN)-2)[:, None], 0)).astype(np.int32, copy=False)

            # readout mutation (keep mild; objective is score_logits)
            if np.random.rand() < 0.35 * mutation_rate:
                childw += (np.random.randn(10, LAST_K) * 0.02).astype(np.float32)
            if np.random.rand() < 0.35 * mutation_rate:
                childb += (np.random.randn(10) * 0.02).astype(np.float32)
            if np.random.rand() < 0.03 * mutation_rate:
                childw += (np.random.randn(10, LAST_K) * 0.10).astype(np.float32)
            if np.random.rand() < 0.35 * mutation_rate:
                childw += (pop_w[np.random.randint(0, POP)] - pop_w[np.random.randint(0, POP)]).astype(np.float32) * np.random.uniform(0, 1.5)
            if np.random.rand() < 0.35 * mutation_rate:
                childb += (pop_b[np.random.randint(0, POP)] - pop_b[np.random.randint(0, POP)]).astype(np.float32) * np.random.uniform(0, 1.5)

            new_g1.append(child1)
            new_g2.append(child2)
            new_w.append(childw.astype(np.float32, copy=False))
            new_b.append(childb.astype(np.float32, copy=False))

    # build next generation (KEEP ALIGNMENT)
    perm = np.random.permutation(len(new_g1))[:POP]
    pop_g1[:] = np.stack([new_g1[i] for i in perm], axis=0).astype(np.int32, copy=False)
    pop_g2[:] = np.stack([new_g2[i] for i in perm], axis=0).astype(np.int32, copy=False)
    pop_w[:]  = np.stack([new_w[i]  for i in perm], axis=0).astype(np.float32, copy=False)
    pop_b[:]  = np.stack([new_b[i]  for i in perm], axis=0).astype(np.float32, copy=False)

    # elite injection (also inject W,b)
    if len(elites_g1) > 0:
        inject = min(100, len(elites_g1))
        #half = inject // 2

        for k in range(inject):
            r = np.random.randint(0, POP)
            src = k
            pop_g1[r] = elites_g1[src].copy()
            pop_g2[r] = elites_g2[src].copy()
            pop_w[r]  = elites_w[src].copy()
            pop_b[r]  = elites_b[src].copy()

        """if len(elites_g1) >= inject:
            lo = max(0, (len(elites_g1)-inject)//2)
            hi = len(elites_g1) - half
            if hi > lo:
                for k in range(half):
                    r = np.random.randint(0, POP)
                    src = np.random.randint(lo, hi)
                    pop_g1[r] = elites_g1[src].copy()
                    pop_g2[r] = elites_g2[src].copy()
                    pop_w[r]  = elites_w[src].copy()
                    pop_b[r]  = elites_b[src].copy()"""

    if step % 200 == 1:
        try:
            np.savez(
                "dats_fast_readout_score.npz",
                pop_g1=pop_g1, pop_g2=pop_g2, pop_w=pop_w, pop_b=pop_b,
                bestacc=bestacc
            )
            surrogatemodel.save("surrogate_model.pt")
        except Exception:
            pass
    gc.collect()
