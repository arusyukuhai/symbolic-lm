import numpy as np
from numba import njit
from copy import deepcopy
import time
import gc
import warnings
from scipy.stats import spearmanr, rankdata
#from datasets import load_dataset

# Load the high-quality subset
#data = load_dataset("HuggingFaceTB/finemath", "finemath-4plus")
#print(data[0])


def chatterjee_correlation(x, y):
    """
    Chatterjeeの順位相関係数 (ξn) を計算する関数
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    
    # 1. Xの昇順にデータをソートする
    # argsortを使ってインデックスを取得し、Yを並べ替える
    sort_idx = np.argsort(x)
    y_sorted = y[sort_idx]
    
    # 2. Yの順位(ランク)を計算する
    # rankdataはデフォルトで平均順位を返すが、ここでは単純化のためordinalを使う
    # (厳密にはタイの処理が必要だが、概念理解のため簡略化)
    r = rankdata(y_sorted, method='ordinal')
    
    # 3. 隣り合うランクの差の絶対値の総和を計算
    diff_sum = np.sum(np.abs(np.diff(r)))
    
    # 4. 公式に当てはめる
    xi = 1 - (3 * diff_sum) / (n**2 - 1)
    
    return xi

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================
# 1D helper (TT/TT2)
# =========================
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

# =========================
# Stable-ish polynomial "attention"
# (normal equation with powers, solve small system)
# =========================
def _attn_poly_fast(k, v, q, deg: int):
    # Ensure inputs are finite and not too extreme
    k = np.nan_to_num(k, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float64).ravel()
    v = np.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float64).ravel()
    q = np.nan_to_num(q, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float64).ravel()

    # Normalize k and q to range around [-1, 1] for stable polynomial basis
    scale = np.max(np.abs(k)) + 1e-12
    kn = k / scale
    qn = q / scale

    size = deg + 1
    max_pow = 2 * deg

    # P[n, m] = kn[n]^m for m=0..2deg
    with np.errstate(over='ignore', invalid='ignore'):
        P = kn[:, None] ** np.arange(max_pow + 1, dtype=np.float64)
    
    # Ensure P is finite (should be if kn in [-1, 1])
    if not np.all(np.isfinite(P)):
        return np.zeros_like(q, dtype=np.float64)

    # A[i,j] = sum kn^(i+j)
    A = np.empty((size, size), dtype=np.float64)
    for i in range(size):
        A[i] = P[:, i:i+size].sum(axis=0)

    # b[i] = sum v*kn^i
    b = (v[:, None] * P[:, :size]).sum(axis=0)

    # adaptive ridge for stability
    ridge = 1e-8 * np.trace(A) / size if size > 0 else 1e-10
    A = A + np.eye(size, dtype=np.float64) * max(ridge, 1e-10)

    # solve
    try:
        coef = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        try:
            coef = np.linalg.lstsq(A, b, rcond=1e-15)[0]
        except np.linalg.LinAlgError:
            # Last resort: return something safe if it still fails to converge
            return np.zeros_like(q, dtype=np.float64)

    # Evaluate at normalized qn
    Q = qn[:, None] ** np.arange(size, dtype=np.float64)
    return (Q @ coef).astype(np.float64)

def attn_poly3_fast(k, v, q):  return _attn_poly_fast(np.nan_to_num(k), np.nan_to_num(v), np.nan_to_num(q), deg=3)
def attn_poly5_fast(k, v, q):  return _attn_poly_fast(np.nan_to_num(k), np.nan_to_num(v), np.nan_to_num(q), deg=5)
def attn_poly11_fast(k, v, q): return _attn_poly_fast(np.nan_to_num(k), np.nan_to_num(v), np.nan_to_num(q), deg=11)

gg   = lambda x: np.abs(x)**(1/3)  * np.sign(x)
ggg  = lambda x: np.abs(x)**0.2    * np.sign(x)
gggg = lambda x: np.abs(x)**(1/11) * np.sign(x)

# 1/2/3-ary function sets
funcs_1 = [
    lambda x: x + 1,
    lambda x: x - 1,
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
    lambda x: np.concatenate((x[::2], x[1::2])),
    lambda x: np.concatenate((x[1::2], x[::2])),
    lambda x: np.concatenate((x[::2], x[::2])),
    lambda x: np.concatenate((x[1::2], x[1::2])),
    lambda x: np.mean(x, dtype=np.float64) + x * 0,
    lambda x: np.std(x, dtype=np.float64) + x * 0,
    lambda x: x - np.mean(x, dtype=np.float64),
    lambda x: x + np.mean(x, dtype=np.float64),
    lambda x: x * np.mean(x, dtype=np.float64),
    lambda x: x * np.std(x, dtype=np.float64),
    lambda x: (np.log(np.std(x) + 1e-12)).astype(np.float64) + x * 0,
    lambda x: x * 0.1,
    lambda x: x * 0.01,
    lambda x: x * 0.001,
    lambda x: (x - np.mean(x)) / (np.std(x) + 1e-12),
    lambda x: x / np.mean(x ** 2) ** 0.5,
    lambda x: x * 10,
    lambda x: x ** 3,
    lambda x: np.sin(x * np.pi),
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
    lambda x, y: np.concatenate((x[::2], y[1::2])),
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

def build_T_distribution_1d(
    i0t, i1t, i2t,
    L=64,             # 1Dベクトル長
    repeats=80,        # 1関数あたりの平均化回数（重いなら下げる）
    warmup=5,          # ウォームアップ回数
    power=0.5,         # 元コードの 0.7
    seed=0,
    eps=1e-12,
):
    """
    各関数の平均実行時間を測って、遅いほど選ばれにくい分布Tを返す。
    返り値:
      T: shape (len(i0t)+len(i1t)+len(i2t),)  正規化済み確率
      times: 同shapeで平均秒
    """
    rng = np.random.default_rng(seed)

    def _bench_unary(f):
        # 入力を毎回変える（キャッシュ効果/分岐癖を減らす）
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
            # 形が変わる関数が混ざっても落ちないように丸める
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

    # --- unary ---
    for idx, f in enumerate(i0t):
        try:
            dt = _bench_unary(f)
        except Exception:
            dt = 1e9  # 壊れる関数は極端に重い扱いにして選ばれないように
        times.append(dt)

    # --- binary ---
    for idx, f in enumerate(i1t):
        try:
            dt = _bench_binary(f)
        except Exception:
            dt = 1e9
        times.append(dt)

    # --- ternary ---
    for idx, f in enumerate(i2t):
        try:
            dt = _bench_ternary(f)
        except Exception:
            dt = 1e9
        times.append(dt)

    times = np.asarray(times, dtype=np.float64)

    # 遅いほど確率小: w = 1/(t^power)
    w = 1.0 / (np.maximum(times, eps) ** power)

    # 全部ゼロになった場合の保険
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones_like(w)

    T = (w / w.sum()).astype(np.float64)
    return T, times

T, times = build_T_distribution_1d(i0t, i1t, i2t)
print("function distribution T:", T)

