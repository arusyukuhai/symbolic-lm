import numpy as np
import cv2
import time
import gc
import tqdm
import torchvision
from numba import njit
import warnings
import matplotlib.pyplot as plt
from rs_cgp import exec_grid_rnn_rust

warnings.simplefilter('ignore')

# -----------------------------
# Dataset
# -----------------------------
ds = torchvision.datasets.CIFAR10
trainset = ds(root="data", train=True, download=True)
testset  = ds(root="data", train=False, download=True)

# -----------------------------
# CGP Functions
# -----------------------------
i0__ = [
    lambda x: np.sin(x),
    lambda x: np.cos(x),
    lambda x: np.sin(x * np.pi),
    lambda x: np.cos(x * np.pi),
    lambda x: x * 2,
    lambda x: x * 10,
    lambda x: x * 0.1,
    lambda x: x * 0.5,
    lambda x: x * 0.9,
    lambda x: x + 1,
    lambda x: x - 1,
    lambda x: x + 10,
    lambda x: x - 10,
    lambda x: -x,
    lambda x: x / 2,
    lambda x: np.tanh(x),
    lambda x: x ** 2,
    lambda x: np.abs(x),
    lambda x: np.sqrt(np.abs(x)),
    lambda x: (np.abs(x) ** (1/3)) * np.sign(x),
    lambda x: np.log(np.square(x) + 1e-12),
    lambda x: x,
    lambda x: np.maximum(x, 0),
    lambda x: np.minimum(x, 0),
    lambda x: x + np.sin(x) ** 2,
]

i1__ = [
    lambda x, y: x + y,
    lambda x, y: x - y,
    lambda x, y: x * y,
    lambda x, y: x + y * 0.5,
    lambda x, y: x + y * 0.1,
    lambda x, y: x + y * 0.9,
    lambda x, y: x / (y ** 2 + 1e-12),
    lambda x, y: np.maximum(x, y),
    lambda x, y: np.minimum(x, y),
    lambda x, y: np.sqrt(x ** 2 + y ** 2),
    lambda x, y: (x + y) / 2,
    lambda x, y: (x - y) ** 2,
    lambda x, y: x * np.tanh(y),
    lambda x, y: x * (np.tanh(y) + 1),
    lambda x, y: np.sin(x * np.pi * y),
    lambda x, y: np.cos(x * np.pi * y),
]

i2__ = [
    lambda x, y, z: x + y + z,
    lambda x, y, z: x + y - z,
    lambda x, y, z: x - y - z,
    lambda x, y, z: x * y * z,
    lambda x, y, z: x * (y ** 2 + 1e-12) / (z ** 2 + 1e-12),
    lambda x, y, z: x / (y ** 2 + z ** 2 + 1e-12),
    lambda x, y, z: np.maximum(x, np.maximum(y, z)),
    lambda x, y, z: np.minimum(x, np.minimum(y, z)),
    lambda x, y, z: np.maximum(x, np.minimum(y, z)),
    lambda x, y, z: np.sqrt(x ** 2 + y ** 2 + z ** 2),
    lambda x, y, z: np.sqrt((x - y) ** 2 + (y - z) ** 2 + (z - x) ** 2),
    lambda x, y, z: np.abs(x * y * z) ** (1/3) * np.sign(x * y * z),
    lambda x, y, z: np.abs((x - y) * (y - z) * (z - x)) ** (1/3) * np.sign((x - y) * (y - z) * (z - x)),
    lambda x, y, z: (x + y + z) / 3,
    lambda x, y, z: np.sin(x * np.pi) * (1 + np.tanh(y)) + np.cos(z * np.pi) * (1 - np.tanh(x)),
]

def activeNode(GENE1, GENE2, N_INPUTS, LAST_K):
    MODELLEN = len(GENE1)
    # Global indices: 0 to N_INPUTS-1 (Inputs), N_INPUTS to N_INPUTS+MODELLEN-1 (Nodes)
    # Output nodes are the last LAST_K nodes
    I = np.arange(N_INPUTS + MODELLEN - LAST_K, N_INPUTS + MODELLEN)
    Nodes = set(I)
    ANodes = list(I)
    while len(ANodes) > 0:
        BNodes = set()
        for N in ANodes:
            if N < N_INPUTS: continue
            local_N = N - N_INPUTS
            f_id = GENE2[local_N, 0]
            
            # Input 1 is always used
            BNodes.add(GENE1[local_N, 0])
            # Input 2 is used for i1 and i2 functions
            if f_id >= len(i0__):
                BNodes.add(GENE1[local_N, 1])
            # Input 3 is used for i2 functions
            if f_id >= len(i0__) + len(i1__):
                BNodes.add(GENE1[local_N, 2])
        
        BNodes = BNodes - Nodes
        Nodes |= BNodes
        ANodes = list(BNodes)
    # Return only node indices that need computation, sorted
    return list(np.sort([N for N in Nodes if N >= N_INPUTS]))

# -----------------------------
# Function speed sampling -> T
# -----------------------------
def build_T_distribution(rounds=40):
    print("Building T distribution (speed-based)...")
    G = []
    # Test tensor for speed sampling
    t2 = np.random.normal(0, 1, (IMG_SCALE, IMG_SCALE)).astype(np.float32)
    
    for f in i0__:
        g0 = time.perf_counter()
        for _ in range(rounds): f(t2)
        G.append((time.perf_counter() - g0) / rounds)
        
    for f in i1__:
        g0 = time.perf_counter()
        for _ in range(rounds): f(t2, t2)
        G.append((time.perf_counter() - g0) / rounds)
        
    for f in i2__:
        g0 = time.perf_counter()
        for _ in range(rounds): f(t2, t2, t2)
        G.append((time.perf_counter() - g0) / rounds)
        
    G = np.asarray(G, dtype=np.float64)
    T = 1.0 / np.maximum(G, 1e-12)
    T = T / T.sum()
    return T

# (execGridRNN and execNode are now replaced by rs_cgp.exec_grid_rnn_rust)

# ============================================================
# POPULATION & EVOLUTION
# ============================================================
MODELLEN = 10000
POP = 50
HIDDEN_DIM = 32
FEATURES_DIM = 10
IMG_SCALE = 32

# CGP Inputs: 3 (RGB) + 2 * HIDDEN_DIM
# CGP Outputs: FEATURES_DIM + HIDDEN_DIM
N_INPUTS = 3 + 2 * HIDDEN_DIM
LAST_K = FEATURES_DIM + HIDDEN_DIM
NUM_FUNCS = len(i0__) + len(i1__) + len(i2__)
T = build_T_distribution(rounds=20)
print("T built. Nonzero ratio:", float((T > 0).mean()))

pop_g1 = np.empty((POP, MODELLEN, 3), dtype=np.int32)
pop_g2 = np.empty((POP, MODELLEN, 1), dtype=np.int32)
pop_w  = (np.random.randn(POP, 10, FEATURES_DIM) * 0.1).astype(np.float32)
pop_b  = np.zeros((POP, 10), dtype=np.float32)

def init_pop():
    for p in range(POP):
        for n in range(MODELLEN):
            pop_g1[p, n] = np.random.randint(0, max(1, n + N_INPUTS), (3,))
            pop_g2[p, n] = np.random.choice(NUM_FUNCS, p=T)
        pop_w[p] = np.random.randn(10, FEATURES_DIM) * 0.1
        pop_b[p] = 0.0

init_pop()

# Elite tracking
elites_g1 = []
elites_g2 = []
elites_w = []
elites_b = []
best_scores_memory = np.zeros(1000)

def score_logits(logits, label):
    order = np.argsort(-logits, axis=1)
    top1 = (order[:, 0] == label).astype(np.float32)
    weights = 1.0 / (np.arange(10) + 1.0) ** 5 # Steeper weights
    hit = (order == label).astype(np.float32) @ weights
    return top1, hit

def train_slice_indices(step, total, slices=1000):
    a = (step % slices) * total // slices
    b = (step % slices + 1) * total // slices
    return a, b

idhk = 0.0

# ============================================================
# MAIN LOOP
# ============================================================
best_acc = 0.0

for step in range(100000):
    losses = np.zeros(POP)
    accs = np.zeros(POP)
    
    # Evaluate on a small batch
    batch_size = 25
    indices = range(*train_slice_indices(step, len(trainset), slices=1000))
    
    # Precompute active nodes for all in population (could be optimized)
    pop_active = [activeNode(pop_g1[p], pop_g2[p], N_INPUTS, LAST_K) for p in range(POP)]
    
    for idx in tqdm.tqdm(indices) if step == 0 else indices:
        img, label = trainset[idx]
        img = np.array(img).astype(np.float32) / 255.0
        img = cv2.resize(img, (IMG_SCALE, IMG_SCALE))
        
        for p in range(POP):
            # Run Grid RNN (Cython)
            active_nodes_arr = np.array(pop_active[p], dtype=np.int32)
            img_hwc = img.astype(np.float32)
            if img_hwc.ndim == 2:
                img_hwc = img_hwc[:, :, None] # Add channel dim if grayscale
                
            feats_map = exec_grid_rnn_rust(
                pop_g1[p], 
                pop_g2[p], 
                active_nodes_arr, 
                img_hwc, 
                HIDDEN_DIM, 
                FEATURES_DIM
            )
            # GAP
            feat = np.mean(feats_map, axis=(0, 1))
            # Readout
            logits = pop_w[p] @ feat + pop_b[p]
            
            t1, hit = score_logits(logits[None, :], label)
            accs[p] += t1[0]
            losses[p] += hit[0]
            
    losses /= batch_size
    accs = (accs / batch_size) * 100.0
    
    rank = np.argsort(-losses) # Highest score first
    if(step == 0):
        idhk = losses[rank[0]]
    idhk = idhk * 0.99 + losses[rank[0]] * 0.01
    # Elite injection/tracking
    slot = step % 1000
    if losses[rank[0]] > best_scores_memory[slot]:
        best_scores_memory[slot] = losses[rank[0]]
        elites_g1.append(pop_g1[rank[0]].copy())
        elites_g2.append(pop_g2[rank[0]].copy())
        elites_w.append(pop_w[rank[0]].copy())
        elites_b.append(pop_b[rank[0]].copy())

    print(f"{step}, {losses[rank[0]] * 100}, {accs[rank[0]]}, {float(100 * np.sum(best_scores_memory) / min(step+1, len(best_scores_memory)))}")
    
    # Evolution: REPRODUCTION
    KEEP = 5
    new_g1 = []
    new_g2 = []
    new_w = []
    new_b = []
    
    # Keeps
    for i in range(KEEP):
        idx = rank[i]
        new_g1.append(pop_g1[idx].copy())
        new_g2.append(pop_g2[idx].copy())
        new_w.append(pop_w[idx].copy())
        new_b.append(pop_b[idx].copy())
        
    mutation_rate = np.random.uniform(0.1, 1.0)
    if np.random.rand() < 0.05:
        mutation_rate *= (np.random.normal(0, 1)**2 + 1)

    while len(new_g1) < POP:
        pa = rank[np.random.randint(0, KEEP)]
        pb = rank[np.random.randint(0, POP)]
        
        cg1 = pop_g1[pa].copy()
        cg2 = pop_g2[pa].copy()
        cw  = pop_w[pa].copy()
        cb  = pop_b[pa].copy()
        
        # Crossover
        if np.random.rand() < 0.5:
            pos1 = np.random.randint(0, MODELLEN-1)
            pos2 = np.random.randint(pos1+1, MODELLEN)
            cg1[pos1:pos2] = pop_g1[pb, pos1:pos2]
            cg2[pos1:pos2] = pop_g2[pb, pos1:pos2]
            
        # Graph Mutations
        if np.random.rand() < 0.1 * mutation_rate:
            for _ in range(np.random.randint(1, 2 ** np.random.randint(1, 12))):
                pos = np.random.randint(0, MODELLEN)
                cg1[pos] = np.random.randint(0, max(1, pos + N_INPUTS), (3,))
        
        if np.random.rand() < 0.1 * mutation_rate:
            for _ in range(np.random.randint(1, 2 ** np.random.randint(1, 12))):
                pos = np.random.randint(0, MODELLEN)
                cg2[pos] = np.random.choice(NUM_FUNCS, p=T)

        # FFT Mix (Structural mutation)
        if np.random.rand() < 0.01 * mutation_rate:
            try:
                tmp = np.floor(np.fft.ifft(
                    np.fft.fft(cg1.astype(np.float32), axis=0) * 
                    np.fft.fft(pop_g1[pb].astype(np.float32), axis=0) /
                    (np.fft.fft(pop_g1[rank[np.random.randint(0, KEEP)]].astype(np.float32), axis=0) + 1e-9),
                    axis=0
                ).real).astype(np.int32)
                cg1 = tmp
            except:
                pass

        # Clamp DAG
        cg1 = np.maximum(0, cg1)
        cg1 = np.minimum(cg1, (np.arange(MODELLEN) + N_INPUTS - 1)[:, None]).astype(np.int32)

        # Readout Mutation
        if np.random.rand() < 0.5 * mutation_rate:
            cw += np.random.randn(*cw.shape).astype(np.float32) * 0.02
        if np.random.rand() < 0.1 * mutation_rate:
            cw += (pop_w[rank[np.random.randint(0, KEEP)]] - pop_w[rank[np.random.randint(0, POP)]]) * np.random.uniform(0, 1.0)
            
        new_g1.append(cg1)
        new_g2.append(cg2)
        new_w.append(cw)
        new_b.append(cb)
        
    pop_g1 = np.array(new_g1)
    pop_g2 = np.array(new_g2)
    pop_w  = np.array(new_w)
    pop_b  = np.array(new_b)
    
    # Elite injection
    if len(elites_g1) > 0 and step % 10 == 0:
        for _ in range(2):
            r = np.random.randint(KEEP, POP)
            idx = np.random.randint(0, len(elites_g1))
            pop_g1[r] = elites_g1[idx].copy()
            pop_g2[r] = elites_g2[idx].copy()
            pop_w[r] = elites_w[idx].copy()
            pop_b[r] = elites_b[idx].copy()

    gc.collect()
