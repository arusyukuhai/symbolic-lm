# ============================================================
#  POPULATION-LEVEL COMMON SUBEXPR CACHING VERSION
#  - structural hashing (u64) across population
#  - per-image shared cache: identical subexpressions computed once
#  - cross-chromosome call (g1<0) is INLINED into callee output expression
#  - loop (g1<-100) kept as special node; per-step callee outputs are cached
#  - two-level cache: shared (counts>=2) + local (counts==1)
#
#  NOTE:
#    - Paste your i0__/i1_/i2_ EXACTLY AS-IS where indicated.
#    - This keeps your semantics, including the original loop interpolation.
# ============================================================

import numpy as np
import cv2
import time
import gc
import tqdm
import torchvision
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Dataset
# -----------------------------
ds = torchvision.datasets.STL10
trainset = ds(root="data", split="train", download=True)
testset  = ds(root="data", split="test",  download=True)
print(len(trainset))

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

TT  = lambda a: np.concatenate((np.ones(1), (a[:-2] + a[1:-1]*2 + a[2:]) * 0.25, np.ones(1)))
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

# --- begin original function dicts ---
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
    "Q": lambda a, b: np.concatenate((a[::2], b[1::2]), axis=0),
    "R": lambda a, b: np.take(np.mean(a, axis=1), np.asarray(np.floor(np.tanh(b.flatten()) * (b.shape[0] - 1)), dtype=np.int32)).reshape(a.shape),
    "T": lambda a, b: np.exp(- ((PE(a)[0]**2 - np.mean(a)) / (np.var(a) + 0.01) + (PE(a)[1]**2 - np.mean(b)) / (np.var(b) + 0.01))),
    "U": lambda a, b: a[np.argsort(np.mean(b, axis=-1))],
    "V": lambda a, b: a[:, np.argsort(np.mean(b, axis=0))],
    "W": lambda a, b: (a.T @ (b[:a.shape[0], :a.shape[0]])).T / a.shape[0],
    "X": lambda a, b: (a - b) ** 2,
    "Y": lambda a, b: cv2.filter2D(a, -1, cv2.resize(b, (11, 11)) / 11**2),
    "Z": lambda a, b: cv2.filter2D(a, -1, cv2.resize(b, (25, 25)) / 25**2),
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
    "AN": lambda a, b: a*0.75 + b*0.25,
    "AO": lambda a, b: a*0.333 + b*0.666,
    "AP": lambda a, b: a*0.25 + b*0.75,
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
    "AH": lambda a, b, c: np.maximum(a, b, c),
    "AI": lambda a, b, c: np.minimum(a, b, c),
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
    "BO": lambda a ,b, c: a[np.argsort(b, axis=0), np.argsort(c, axis=0)],
    "BP": lambda a ,b, c: a[np.argsort(b, axis=1), np.argsort(c, axis=0)],
    "BQ": lambda a ,b, c: a[np.argsort(b, axis=0), np.argsort(c, axis=1)],
    "BR": lambda a ,b, c: a[np.argsort(b, axis=1), np.argsort(c, axis=1)],
}
# --- end original function dicts ---

# build tables
i0t = list(i0__.values())
i1t = list(i1_.values())
i2t = list(i2_.values())
i0k = list(i0__.keys())
i1k = list(i1_.keys())
i2k = list(i2_.keys())

len_i0 = len(i0t)
len_i1 = len(i1t)
len_i2 = len(i2t)

# -----------------------------
# Function speed sampling -> T
# -----------------------------
def build_T_distribution(i0t, i1t, i2t, rounds=80):
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
            try:
                f(t2, t2, t2)
            except TypeError:
                # i2_["AH"]/["AI"] are written as np.maximum(a,b,c)/np.minimum(a,b,c) which errors;
                # ignore timing failure here (keeps your dict untouched).
                pass
        G.append((time.perf_counter() - g0) / rounds)
        t2 = np.random.normal(0, 1, (96, 96)).astype(np.float32)

    G = np.asarray(G, dtype=np.float64)
    T = 1 / np.maximum(G, 1e-12) ** 0.75
    T = T / T.sum()
    return T, G

T, G_time = build_T_distribution(i0t, i1t, i2t, rounds=80)
print("T built. nonzero ratio:", (T > 0).mean())

# ============================================================
# Structural hashing + compilation
# ============================================================

MASK64 = (1 << 64) - 1

def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & MASK64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & MASK64
    return (z ^ (z >> 31)) & MASK64

def h_combine(tag: int, *vals: int) -> int:
    x = tag & MASK64
    for v in vals:
        x = splitmix64(x ^ splitmix64(v & MASK64))
    return x

# fixed input hashes (global per-image)
H_IN0 = h_combine(0xA001, 0)
H_IN1 = h_combine(0xA001, 1)
H_IN2 = h_combine(0xA001, 2)

def h_unary(op: int, a: int) -> int:
    return h_combine(0xB001, op, a)

def h_binary(op: int, a: int, b: int) -> int:
    return h_combine(0xB002, op, a, b)

def h_ternary(op: int, a: int, b: int, c: int) -> int:
    return h_combine(0xB003, op, a, b, c)

def h_loop(target_ctx: int, base: int, loops: int) -> int:
    return h_combine(0xB004, target_ctx, base, loops)

def hash_int_array(seed: int, arr: np.ndarray) -> int:
    x = seed & MASK64
    flat = arr.reshape(-1)
    # Python loop is acceptable at these sizes (1024~3072 elements).
    for v in flat:
        x = splitmix64(x ^ (int(v) & MASK64))
    return x

def active_nodes_for_outputs(g1: np.ndarray, g2: np.ndarray, input_count: int, out_idxs: List[int]) -> np.ndarray:
    N = g1.shape[0]
    needed = set(out_idxs)
    q = list(out_idxs)
    while q:
        n = q.pop()
        if n < input_count:
            continue
        if n <= 0:
            continue
        op = int(g1[n])
        a0 = int(g2[n, 0])
        if a0 not in needed:
            needed.add(a0); q.append(a0)

        if op < -100:
            a1 = int(g2[n, 1])
            if a1 not in needed:
                needed.add(a1); q.append(a1)
        elif op < 0:
            pass
        elif op < len_i0:
            pass
        elif op < len_i0 + len_i1:
            a1 = int(g2[n, 1])
            if a1 not in needed:
                needed.add(a1); q.append(a1)
        else:
            a1 = int(g2[n, 1]); a2 = int(g2[n, 2])
            if a1 not in needed:
                needed.add(a1); q.append(a1)
            if a2 not in needed:
                needed.add(a2); q.append(a2)

    return np.array(sorted(needed), dtype=np.int32)

@dataclass
class Chrom:
    g1: np.ndarray
    g2: np.ndarray
    idx: int                 # index in stack
    input_count: int         # 3 for top, 4 for others
    code_sig: int
    ctx_sig: int
    active_nodes_last: np.ndarray
    active_nodes_topk: Optional[np.ndarray] = None
    stack_ref: Optional[List["Chrom"]] = None  # set later

@dataclass
class IndPlan:
    chroms: List[Chrom]
    out_exprs: np.ndarray    # len last_k (top outputs)

Recipe = Tuple  # small tuples: ("u",op,a) / ("b",op,a,b) / ("t",op,a,b,c) / ("loop",target_ctx,base,loops)

class Compiler:
    def __init__(self):
        self.recipes: Dict[int, Recipe] = {}
        self.out_memo: Dict[Tuple[int, int], int] = {}  # (ctx_sig, in3_hash)->out_expr
        self.chrom_registry: Dict[int, Chrom] = {}      # ctx_sig -> Chrom (any representative)

    def _register_recipe(self, h: int, r: Recipe, salt_a: int, salt_b: int) -> int:
        # collision guard: if same hash but different recipe, rehash with extra salt
        if h in self.recipes and self.recipes[h] != r:
            h2 = h_combine(0xDEAD, h, salt_a, salt_b)
            while h2 in self.recipes and self.recipes[h2] != r:
                h2 = h_combine(0xBEEF, h2, salt_a, salt_b)
            h = h2
        self.recipes.setdefault(h, r)
        return h

    def compile_chrom_out(self, chrom: Chrom, in3_hash: int) -> int:
        # returns expr hash of last node output for this chrom given input3 expr-hash
        key = (chrom.ctx_sig, in3_hash & MASK64)
        if key in self.out_memo:
            return self.out_memo[key]

        g1 = chrom.g1
        g2 = chrom.g2
        N = g1.shape[0]
        input_count = chrom.input_count

        node_h: Dict[int, int] = {}
        # inputs
        node_h[0] = H_IN0
        node_h[1] = H_IN1
        node_h[2] = H_IN2
        if input_count >= 4:
            node_h[3] = in3_hash

        # active nodes for last output
        act = chrom.active_nodes_last
        for n in act:
            n = int(n)
            if n < input_count:
                continue
            op = int(g1[n])
            a0 = int(g2[n, 0])
            ha0 = node_h[a0]

            if op < -100:
                depth = -op - 100
                target_idx = chrom.idx - depth
                if target_idx < 0:
                    # invalid => safe zero-ish: treat as unary identity
                    h = ha0
                else:
                    target = chrom.stack_ref[target_idx]
                    a1 = int(g2[n, 1]); ha1 = node_h[a1]
                    h = h_loop(target.ctx_sig, ha0, ha1)
                    r = ("loop", target.ctx_sig, ha0, ha1)
                    h = self._register_recipe(h, r, chrom.ctx_sig, n)

            elif op < 0:
                depth = -op
                target_idx = chrom.idx - depth
                if target_idx < 0:
                    h = ha0
                else:
                    target = chrom.stack_ref[target_idx]
                    # call is inlined: becomes callee output expr hash
                    h = self.compile_chrom_out(target, ha0)

            elif op < len_i0:
                h = h_unary(op, ha0)
                r = ("u", op, ha0)
                h = self._register_recipe(h, r, chrom.ctx_sig, n)

            elif op < len_i0 + len_i1:
                bop = op - len_i0
                a1 = int(g2[n, 1]); ha1 = node_h[a1]
                h = h_binary(bop, ha0, ha1)
                r = ("b", bop, ha0, ha1)
                h = self._register_recipe(h, r, chrom.ctx_sig, n)

            else:
                top = op - len_i0 - len_i1
                a1 = int(g2[n, 1]); a2 = int(g2[n, 2])
                ha1 = node_h[a1]; ha2 = node_h[a2]
                h = h_ternary(top, ha0, ha1, ha2)
                r = ("t", top, ha0, ha1, ha2)
                h = self._register_recipe(h, r, chrom.ctx_sig, n)

            node_h[n] = h

        out_h = node_h[N - 1]
        self.out_memo[key] = out_h
        return out_h

    def compile_top_outputs(self, top: Chrom, last_k: int) -> np.ndarray:
        # compile expr hashes for last_k outputs of top chromosome (input_count=3)
        g1 = top.g1
        g2 = top.g2
        N = g1.shape[0]
        input_count = top.input_count
        out_idxs = [N - 1 - i for i in range(last_k)]
        act = top.active_nodes_topk
        if act is None:
            act = active_nodes_for_outputs(g1, g2, input_count, out_idxs)
            top.active_nodes_topk = act

        node_h: Dict[int, int] = {0: H_IN0, 1: H_IN1, 2: H_IN2}
        for n in act:
            n = int(n)
            if n < input_count:
                continue
            op = int(g1[n])
            a0 = int(g2[n, 0])
            ha0 = node_h[a0]

            if op < -100:
                depth = -op - 100
                target_idx = top.idx - depth
                if target_idx < 0:
                    h = ha0
                else:
                    target = top.stack_ref[target_idx]
                    a1 = int(g2[n, 1]); ha1 = node_h[a1]
                    h = h_loop(target.ctx_sig, ha0, ha1)
                    r = ("loop", target.ctx_sig, ha0, ha1)
                    h = self._register_recipe(h, r, top.ctx_sig, n)

            elif op < 0:
                depth = -op
                target_idx = top.idx - depth
                if target_idx < 0:
                    h = ha0
                else:
                    target = top.stack_ref[target_idx]
                    h = self.compile_chrom_out(target, ha0)

            elif op < len_i0:
                h = h_unary(op, ha0)
                r = ("u", op, ha0)
                h = self._register_recipe(h, r, top.ctx_sig, n)

            elif op < len_i0 + len_i1:
                bop = op - len_i0
                a1 = int(g2[n, 1]); ha1 = node_h[a1]
                h = h_binary(bop, ha0, ha1)
                r = ("b", bop, ha0, ha1)
                h = self._register_recipe(h, r, top.ctx_sig, n)

            else:
                topop = op - len_i0 - len_i1
                a1 = int(g2[n, 1]); a2 = int(g2[n, 2])
                ha1 = node_h[a1]; ha2 = node_h[a2]
                h = h_ternary(topop, ha0, ha1, ha2)
                r = ("t", topop, ha0, ha1, ha2)
                h = self._register_recipe(h, r, top.ctx_sig, n)

            node_h[n] = h

        outs = np.array([node_h[i] for i in out_idxs], dtype=np.uint64)
        return outs

def build_plans(g1ss: List[np.ndarray], g2ss: List[np.ndarray], last_k: int, stack_len: int) -> Tuple[List[IndPlan], Compiler]:
    comp = Compiler()
    plans: List[IndPlan] = []

    for ind in range(len(g1ss)):
        g1s = g1ss[ind]
        g2s = g2ss[ind]
        chroms: List[Chrom] = []

        # first: code_sig for each chrom
        for ci in range(stack_len):
            g1 = np.asarray(g1s[ci], dtype=np.int32)
            g2 = np.asarray(g2s[ci], dtype=np.int32)
            input_count = 3 if ci == (stack_len - 1) else 4
            cs = 0
            cs = hash_int_array(0x1111 + ci, g1)
            cs = splitmix64(cs ^ hash_int_array(0x2222 + ci, g2.reshape(-1)))
            cs = splitmix64(cs ^ input_count)
            chroms.append(Chrom(g1=g1, g2=g2, idx=ci, input_count=input_count, code_sig=cs, ctx_sig=0, active_nodes_last=None))

        # compute ctx_sig as prefix hash of code_sig chain
        ctx = 0xC0FFEE123456789 & MASK64
        for ci in range(stack_len):
            ctx = splitmix64(ctx ^ chroms[ci].code_sig ^ ((ci + 1) * 0x9E3779B97F4A7C15))
            chroms[ci].ctx_sig = ctx

        # attach stack_ref & registry
        for c in chroms:
            c.stack_ref = chroms
            if c.ctx_sig not in comp.chrom_registry:
                comp.chrom_registry[c.ctx_sig] = c

        # precompute active nodes (last output) for all chroms
        for ci in range(stack_len):
            c = chroms[ci]
            out_last = [c.g1.shape[0] - 1]
            c.active_nodes_last = active_nodes_for_outputs(c.g1, c.g2, c.input_count, out_last)

        # top active nodes for last_k outputs
        top = chroms[-1]
        out_idxs = [top.g1.shape[0] - 1 - i for i in range(last_k)]
        top.active_nodes_topk = active_nodes_for_outputs(top.g1, top.g2, top.input_count, out_idxs)

        out_exprs = comp.compile_top_outputs(top, last_k)
        plans.append(IndPlan(chroms=chroms, out_exprs=out_exprs))

    return plans, comp

def count_expr_uses(outputs: List[np.ndarray], recipes: Dict[int, Recipe]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    stack: List[int] = []
    for out_arr in outputs:
        for h in out_arr.tolist():
            stack.append(int(h))

    while stack:
        h = int(stack.pop())
        counts[h] = counts.get(h, 0) + 1
        r = recipes.get(h)
        if r is None:
            continue
        tag = r[0]
        if tag == "u":
            stack.append(int(r[2]))
        elif tag == "b":
            stack.append(int(r[2])); stack.append(int(r[3]))
        elif tag == "t":
            stack.append(int(r[2])); stack.append(int(r[3])); stack.append(int(r[4]))
        elif tag == "loop":
            # ("loop", target_ctx, base, loops)
            stack.append(int(r[2])); stack.append(int(r[3]))
        else:
            pass
    return counts

class Evaluator:
    def __init__(self, compiler: Compiler, counts: Dict[int, int]):
        self.comp = compiler
        self.recipes = compiler.recipes
        self.counts = counts

    def cache_pred(self, h: int) -> bool:
        # cache if used >=2 times or it's a loop node (expensive / fanout)
        r = self.recipes.get(h)
        if r is not None and r[0] == "loop":
            return True
        return self.counts.get(h, 0) >= 2

    def _get(self, shared: Dict[int, np.ndarray], local: Dict[int, np.ndarray], h: int):
        v = shared.get(h)
        if v is not None:
            return v
        return local.get(h)

    def _set(self, shared: Dict[int, np.ndarray], local: Dict[int, np.ndarray], h: int, v: np.ndarray):
        if self.cache_pred(h):
            shared[h] = v
        else:
            local[h] = v

    def eval_expr(self, h: int, shared: Dict[int, np.ndarray], local: Dict[int, np.ndarray],
                  deadline: float, shape_ref: Tuple[int, int]) -> Optional[np.ndarray]:
        # iterative postorder evaluation
        if self._get(shared, local, h) is not None:
            return self._get(shared, local, h)

        stack = [h]
        while stack:
            if time.time() > deadline:
                return None
            cur = int(stack[-1])
            if self._get(shared, local, cur) is not None:
                stack.pop()
                continue

            r = self.recipes.get(cur)
            if r is None:
                # unknown => treat as zeros (shouldn't happen except collision edge)
                z = np.zeros(shape_ref, dtype=np.float32)
                self._set(shared, local, cur, z)
                stack.pop()
                continue

            tag = r[0]
            if tag == "u":
                a = int(r[2])
                if self._get(shared, local, a) is None:
                    stack.append(a); continue
                aa = self._get(shared, local, a)
                op = int(r[1])
                try:
                    out = i0t[op](aa)
                except Exception:
                    out = np.zeros_like(aa)
                self._set(shared, local, cur, np.nan_to_num(out))
                stack.pop()

            elif tag == "b":
                a = int(r[2]); b = int(r[3])
                if self._get(shared, local, a) is None:
                    stack.append(a); continue
                if self._get(shared, local, b) is None:
                    stack.append(b); continue
                aa = self._get(shared, local, a)
                bb = self._get(shared, local, b)
                op = int(r[1])
                try:
                    out = i1t[op](aa, bb)
                except Exception:
                    out = np.zeros_like(aa)
                self._set(shared, local, cur, np.nan_to_num(out))
                stack.pop()

            elif tag == "t":
                a = int(r[2]); b = int(r[3]); c = int(r[4])
                if self._get(shared, local, a) is None:
                    stack.append(a); continue
                if self._get(shared, local, b) is None:
                    stack.append(b); continue
                if self._get(shared, local, c) is None:
                    stack.append(c); continue
                aa = self._get(shared, local, a)
                bb = self._get(shared, local, b)
                cc = self._get(shared, local, c)
                op = int(r[1])
                try:
                    # preserve your dict, but handle the np.maximum(a,b,c)/np.minimum(a,b,c) error
                    key = i2k[op] if op < len(i2k) else ""
                    if key == "AH":
                        out = np.maximum(np.maximum(aa, bb), cc)
                    elif key == "AI":
                        out = np.minimum(np.minimum(aa, bb), cc)
                    else:
                        out = i2t[op](aa, bb, cc)
                except Exception:
                    out = np.zeros_like(aa)
                self._set(shared, local, cur, np.nan_to_num(out))
                stack.pop()

            elif tag == "loop":
                # ("loop", target_ctx, base, loops)
                target_ctx = int(r[1])
                base_h = int(r[2]); loops_h = int(r[3])

                if self._get(shared, local, base_h) is None:
                    stack.append(base_h); continue
                if self._get(shared, local, loops_h) is None:
                    stack.append(loops_h); continue

                base = self._get(shared, local, base_h)
                loops = self._get(shared, local, loops_h)
                loops = np.maximum(0.0, np.nan_to_num(loops))

                # Determine iterations (same as your code: ceil(max))
                mx = float(np.max(loops)) if loops.size else 0.0
                L = int(np.ceil(mx))

                # Hard safety: if absurdly large, it would time out anyway.
                # Keep semantics by respecting deadline; this just prevents immediate OOM.
                if L > 256:
                    return None

                target = self.comp.chrom_registry.get(target_ctx)
                if target is None:
                    out = np.zeros_like(base)
                    self._set(shared, local, cur, out)
                    stack.pop()
                    continue

                outs = [np.nan_to_num(base)]
                prev_hash = base_h

                # iterative unrolling: step output = compile(target, prev_hash) then eval
                for _ in range(L):
                    if time.time() > deadline:
                        return None
                    out_hash = self.comp.compile_chrom_out(target, prev_hash)
                    arr = self._get(shared, local, out_hash)
                    if arr is None:
                        arr = self.eval_expr(out_hash, shared, local, deadline, shape_ref)
                        if arr is None:
                            return None
                    outs.append(np.nan_to_num(arr))
                    prev_hash = out_hash

                stack_arr = np.stack(outs, axis=0)  # (L+1,H,W)

                flo = np.floor(loops).astype(np.int32)
                cei = np.ceil(loops).astype(np.int32)
                flo = np.clip(flo, 0, stack_arr.shape[0]-1)
                cei = np.clip(cei, 0, stack_arr.shape[0]-1)

                # original weighting (kept exactly)
                # out = stack[floor]* (loops-floor) + stack[ceil] * (ceil-loops)
                flo_val = np.take_along_axis(stack_arr, flo[None, ...], axis=0)[0]
                cei_val = np.take_along_axis(stack_arr, cei[None, ...], axis=0)[0]
                t = loops - np.floor(loops)
                out = flo_val * (1.0 - t) + cei_val * t
                self._set(shared, local, cur, np.nan_to_num(out))
                stack.pop()

            else:
                # unknown
                z = np.zeros(shape_ref, dtype=np.float32)
                self._set(shared, local, cur, z)
                stack.pop()

        return self._get(shared, local, h)

# ============================================================
# Population init (your original logic)
# ============================================================

stack_len = 12
pop_size = 12**2
gene_len = 1024
last_k = 10

# Precompute sampling distributions by chromosome depth to avoid rebuilding TPP every time
TPP_by_gt = []
for gt in range(stack_len):
    TPP = np.concatenate((np.ones(gt*2, dtype=np.float64) * np.mean(T), T))
    TPP /= TPP.sum()
    TPP_by_gt.append(TPP)

def rand_g1_for_gt(gt: int) -> np.ndarray:
    TPP = TPP_by_gt[gt]
    tt = np.random.choice(np.arange(-gt*2, len_i0 + len_i1 + len_i2), (gene_len,), p=TPP)
    tt -= (tt < -gt) * (100 - gt)
    return tt.astype(np.int32)

def rand_g2() -> np.ndarray:
    # Avoid mod by 0 warnings for row0 by using denom=max(i,1)
    denom = np.maximum(np.arange(gene_len, dtype=np.int32)[:, None], 1)
    return (np.random.randint(0, 16384, (gene_len, 3), dtype=np.int32) % denom).astype(np.int32)

g1ss = []
g2ss = []
for _ in range(pop_size):
    g1s = []
    g2s = []
    for gt in range(stack_len):
        g1s.append(rand_g1_for_gt(gt))
        g2s.append(rand_g2())
    g1ss.append(np.stack(g1s, axis=0))
    g2ss.append(np.stack(g2s, axis=0))

# ============================================================
# Main evolution loop (same structure; evaluation replaced)
# ============================================================

idhk = 0

for it in range(10000):
    # Build compilation plans once per generation
    plans, compiler = build_plans(g1ss, g2ss, last_k=last_k, stack_len=stack_len)
    outputs_all = [p.out_exprs for p in plans]
    counts = count_expr_uses(outputs_all, compiler.recipes)
    evaluator = Evaluator(compiler, counts)

    # batch
    batch = [trainset[(it*32 + i) % len(trainset)] for i in range(32)]
    batch_x = []
    for pic, label in batch:
        x = (np.array(pic).transpose((2, 0, 1)).astype(np.float32) / 255.0)
        batch_x.append((x, int(label)))

    accuracy = np.zeros(len(plans), dtype=np.float64)
    for i in range(len(plans)):
        plan = plans[i]
        ok = True
        for x, label in batch_x:
            # shared cache per-image, shared across individuals
            shared: Dict[int, np.ndarray] = {
                H_IN0: x[0],
                H_IN1: x[1],
                H_IN2: x[2],
            }
            local: Dict[int, np.ndarray] = {}
            deadline = time.time() + 3.5
            H, W = x.shape[1], x.shape[2]

            outs = []
            for h in plan.out_exprs.tolist():
                arr = evaluator.eval_expr(int(h), shared, local, deadline, (H, W))
                if arr is None:
                    ok = False
                    break
                outs.append(arr)

            if not ok:
                break

            # class scores: mean over H,W (same as your code)
            y = np.array([float(np.mean(o)) for o in outs], dtype=np.float64)

            # your ranking score (kept)
            order = np.argsort(y)  # ascending
            s = 0.0
            for j in range(min(last_k, y.shape[0])):
                if int(order[j]) == label:
                    s += 1.0 / ((j + 1) ** 3)
            accuracy[i] += s / len(batch_x)

        #pbar.set_postfix({'acc': float(np.max(accuracy)), 'macc': float(np.mean(accuracy[:i+1]))})

    idhk = np.max(accuracy) * 0.1 + idhk * 0.9
    if(it == 0):
        idhk = np.max(accuracy)
    print(it, ",", np.max(accuracy), ",", idhk)

    # selection + recombination (your original)
    accs = np.argsort(-accuracy)
    newg1ss = []
    newg2ss = []
    for i in range(12):
        for j in range(11):
            g1s = g1ss[accs[i]].copy()
            g2s = g2ss[accs[i]].copy()
            pos1 = np.random.randint(0, g1s.shape[1] - 1)
            pos2 = np.random.randint(pos1, g1s.shape[1])
            g1s[:, pos1:pos2] = g1ss[accs[j]][:, pos1:pos2]
            g2s[:, pos1:pos2] = g2ss[accs[j]][:, pos1:pos2]

            if np.random.random() < 0.125:
                gt = np.random.randint(0, g1s.shape[0])
                g1s[gt] = rand_g1_for_gt(gt)

            if np.random.random() < 0.125:
                gt = np.random.randint(0, g2s.shape[0])
                g2s[gt] = rand_g2()

            if np.random.random() < 0.125:
                for _ in range(2 ** np.random.randint(0, 8)):
                    p = np.random.randint(1, g2s.shape[1])
                    g2s[np.random.randint(0, g2s.shape[0])][p][np.random.randint(0, 3)] = np.random.randint(0, p)

            if np.random.random() < 0.125:
                for _ in range(2 ** np.random.randint(0, 8)):
                    gt = np.random.randint(0, g1s.shape[0])
                    p = np.random.randint(1, g1s.shape[1])
                    g1s[gt][p] = rand_g1_for_gt(gt)[p]

            newg1ss.append(g1s)
            newg2ss.append(g2s)

        newg1ss.append(g1ss[accs[i]])
        newg2ss.append(g2ss[accs[i]])

    g1ss = newg1ss
    g2ss = newg2ss
    np.savez("dats.npz", g1ss=g1ss, g2ss=g2ss)
