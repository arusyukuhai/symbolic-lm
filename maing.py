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

# =========================
# Structured execution engine (1D)
# =========================
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
                if a >= 0 and a < MODELLEN and used[ind, a] == 0:
                    stack[top] = a; top += 1
            elif func_id < (len_i0 + len_i1):
                a = int(abs(G1[ind, n, 0]))
                b = int(abs(G1[ind, n, 1]))
                if a >= 0 and a < MODELLEN and used[ind, a] == 0:
                    stack[top] = a; top += 1
                if b >= 0 and b < MODELLEN and used[ind, b] == 0:
                    stack[top] = b; top += 1
            else:
                a = int(abs(G1[ind, n, 0]))
                b = int(abs(G1[ind, n, 1]))
                c = int(abs(G1[ind, n, 2]))
                if a >= 0 and a < MODELLEN and used[ind, a] == 0:
                    stack[top] = a; top += 1
                if b >= 0 and b < MODELLEN and used[ind, b] == 0:
                    stack[top] = b; top += 1
                if c >= 0 and c < MODELLEN and used[ind, c] == 0:
                    stack[top] = c; top += 1

        for j in range(num_inputs):
            used[ind, j] = 1

    return used

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

            if func_id < len_i0:
                a = int(abs(G1[ind, node, 0]))
                child_sid = node_structs[ind, a]
                key = ((1 << 60) | (func_id << 36) | ((child_sid & 0xFFFFF) << 16) | (a_quant & 0xFFFF))

            elif func_id < (len_i0 + len_i1):
                bi = func_id - len_i0
                a = int(abs(G1[ind, node, 0])); b = int(abs(G1[ind, node, 1]))
                child_a = node_structs[ind, a]
                child_b = node_structs[ind, b]
                key = ((2 << 60) | (bi << 36) | ((child_a & 0xFFFFF) << 16) | (child_b & 0xFFF))
                key = key ^ (a_quant & 0xFFFF)

            else:
                ci = func_id - (len_i0 + len_i1)
                a = int(abs(G1[ind, node, 0])); b = int(abs(G1[ind, node, 1])); c = int(abs(G1[ind, node, 2]))
                child_a = node_structs[ind, a]
                child_b = node_structs[ind, b]
                child_c = node_structs[ind, c]
                key = ((3 << 60) | (ci << 36) | ((child_a & 0xFFFFF) << 16) | ((child_b & 0xFFF) << 6) | (child_c & 0x3F))
                key = key ^ (a_quant & 0xFFFF)

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

    # map struct -> nodes (unused here, but kept)
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
    # なるべくコピーしないで float32 1D に揃える
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

def batch_exec_structured_logits_1d(
    x_inputs: np.ndarray,   # (num_inputs, L)
    node_structs,
    struct_type,
    struct_func,
    struct_ch1,
    struct_ch2,
    struct_ch3,
    struct_alpha,
    topo,
    last_k=1,
    restrict=True,
):
    x_inputs = np.asarray(x_inputs, dtype=np.float32)
    if x_inputs.ndim != 2:
        raise ValueError("x_inputs must be (num_inputs, L)")
    num_inputs, L = x_inputs.shape

    N = node_structs.shape[0]
    MODELLEN = node_structs.shape[1]
    S = struct_type.shape[0]

    # --- needed mask (Pythonで十分軽いが、リストappendを減らす) ---
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
                needed[c1] = True; 
                if qlen >= q.size: q = np.resize(q, q.size * 2)
                q[qlen] = c1; qlen += 1
            if c2 >= 0 and not needed[c2]:
                needed[c2] = True; 
                if qlen >= q.size: q = np.resize(q, q.size * 2)
                q[qlen] = c2; qlen += 1
            if c3 >= 0 and not needed[c3]:
                needed[c3] = True; 
                if qlen >= q.size: q = np.resize(q, q.size * 2)
                q[qlen] = c3; qlen += 1
    else:
        needed = None  # 全部必要扱い

    # --- outputs ---
    outputs = [None] * S
    # input sid 0..num_inputs-1
    for i in range(min(num_inputs, S)):
        outputs[i] = x_inputs[i]

    # ローカル参照で少し速く
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
            # alpha mix: (1-a)*base + a*x
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

    # --- logits: last_k の sid だけ集計（np.uniqueをやめる）---
    last_nodes = np.arange(max(0, MODELLEN - last_k), MODELLEN, dtype=np.int32)
    last_sids = node_structs[:, last_nodes]  # (N,last_k)

    need_last = np.zeros(S, dtype=np.bool_)
    for i in range(N):
        for j in range(last_k):
            s = int(last_sids[i, j])
            if s >= 0:
                need_last[s] = True

    sid_mean = np.zeros(S, dtype=np.float32)
    sid_has = np.zeros(S, dtype=np.bool_)
    for s in range(S):
        if not need_last[s]:
            continue
        arr = outputs[s]
        if arr is None:
            continue
        v = float(np.mean(arr))
        if np.isfinite(v):
            sid_mean[s] = v
            sid_has[s] = True

    logits = np.zeros((N, last_k), dtype=np.float32)
    for i in range(N):
        for j in range(last_k):
            s = int(last_sids[i, j])
            if s >= 0 and sid_has[s]:
                logits[i, j] = sid_mean[s]

    return logits

MODELLEN = 65536

for ___ in range(32):
    GENES1, GENES2, GENES3 = [], [], []
    for _ in range(1):
        G1 = np.abs((1 - np.random.uniform(0, 1, (MODELLEN, 3))**1.25) * (np.arange(MODELLEN)[:, None]))
        G2 = np.random.choice(len_i0 + len_i1 + len_i2, size=(MODELLEN,), p=T)
        G3 = np.random.uniform(0, 1, size=(MODELLEN,)).astype(np.float32)
        GENES1.append(G1.astype(np.int64))
        GENES2.append(G2.astype(np.int64))
        GENES3.append(np.zeros_like(G3.astype(np.float32)))

    G1 = np.stack(GENES1, axis=0)
    G2 = np.stack(GENES2, axis=0)
    G3 = np.zeros_like(np.stack(GENES3, axis=0))

    node_structs, struct_type, struct_func, struct_ch1, struct_ch2, struct_ch3, struct_alpha, idxs, pairs = \
        precompute_structs_numba(G1, G2, G3, len_i0, len_i1, len_i2, num_inputs=1, last_k=1)
    topo = topo_sort_structs_numba_from_arrays(struct_type, struct_ch1, struct_ch2, struct_ch3)

    import tqdm

    import matplotlib.pyplot as plt
    xs = []
    ys = []

    g = np.linspace(0,2,1024)
    for x in tqdm.tqdm(np.linspace(-1, 1, 1024)):
        # x_inputs: (16, L)
        logit = batch_exec_structured_logits_1d(
            g[None] * x, node_structs, struct_type, struct_func, struct_ch1, struct_ch2, struct_ch3,
            struct_alpha, topo, last_k=1, restrict=True
        )
        xs.append(x)
        ys.append(np.mean(logit))

    plt.plot(xs, ys)
    plt.show()