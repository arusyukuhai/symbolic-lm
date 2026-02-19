import numpy as np
import cv2
import time
import gc
import tqdm
import torchvision
from numba import njit
import warnings
import matplotlib.pyplot as plt

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
    I = np.arange(len(GENE1) - LAST_K, len(GENE1))
    Nodes = set(I)
    ANodes = list(I)
    while len(ANodes) > 0:
        BNodes = set()
        for N in ANodes:
            if(GENE1[int(N)][0] >= N_INPUTS):
                BNodes.add(GENE1[int(N)][0])
            if(GENE1[int(N)][1] >= N_INPUTS and GENE2[int(N)][0] >= len(i0__)):
                BNodes.add(GENE1[int(N)][1])
            if(GENE1[int(N)][2] >= N_INPUTS and GENE2[int(N)][0] >= len(i0__) + len(i1__)):
                BNodes.add(GENE1[int(N)][2])
        Nodes |= BNodes
        ANodes = list(BNodes)
    return list(np.floor(np.sort(list(Nodes))))

def execNode(GENE1, GENE2, ActiveNodes, Inputs, LAST_K):
    Calculated = {}
    for i, I in enumerate(Inputs):
        Calculated[i] = I
    for N in ActiveNodes:
        if(N < len(Inputs)):
            continue
        if(GENE2[int(N)][0] < len(i0__)):
            Calculated[N] = i0__[GENE2[int(N)][0]](Calculated[GENE1[int(N)][0]])
        elif(GENE2[int(N)][0] < len(i0__) + len(i1__)):  
            Calculated[N] = i1__[GENE2[int(N)][0] - len(i0__)](Calculated[GENE1[int(N)][0]], Calculated[GENE1[int(N)][1]])
        else:
            Calculated[N] = i2__[GENE2[int(N)][0] - len(i0__) - len(i1__)](Calculated[GENE1[int(N)][0]], Calculated[GENE1[int(N)][1]], Calculated[GENE1[int(N)][2]])
    return [Calculated[N] for N in range(len(GENE1) - LAST_K, len(GENE1))]

def execGridRNN(GENE1, GENE2, ActiveNodes, Image, hidden_dim, out_dim):
    H, W = Image.shape[:2]
    C = Image.shape[2] if len(Image.shape) > 2 else 1
    
    # Hidden states: [height, width, hidden_dim]
    # Initialized to zeros
    Hidden = np.zeros((H + 1, W + 1, hidden_dim))
    Output = np.zeros((H, W, out_dim))
    
    # LAST_K = out_dim + hidden_dim
    LAST_K = out_dim + hidden_dim
    N_INPUTS = C + 2 * hidden_dim # current pixel + hidden from top + hidden from left
    
    for y in range(H):
        for x in range(W):
            # Form input vector: [Pixel(s), Top_Hidden, Left_Hidden]
            pixel_val = Image[y, x]
            if np.isscalar(pixel_val):
                pixel_val = [pixel_val]
            else:
                pixel_val = list(pixel_val)
                
            top_h = Hidden[y, x+1]
            left_h = Hidden[y+1, x]
            
            inputs = np.concatenate([pixel_val, top_h, left_h])
            
            # Execute CGP
            res = execNode(GENE1, GENE2, ActiveNodes, inputs, LAST_K)
            
            # res: [out_channel1, out_channel2, ..., new_h1, new_h2, ...]
            Output[y, x] = np.array(res[:out_dim])
            Hidden[y+1, x+1] = np.array(res[out_dim:])
            
    return Output, Hidden[1:, 1:]

# ============================================================
# DEMONSTRATION
# ============================================================
if __name__ == "__main__":
    # Settings
    HIDDEN_DIM = 4
    OUT_DIM = 1 # RGB-like output
    MODELLEN = 1000
    IMG_SIZE = 96
    
    # Inputs: 1 (pixel) + 2 * HIDDEN_DIM (top + left)
    N_INPUTS = 1 + 2 * HIDDEN_DIM
    # Outputs: OUT_DIM (pixels) + HIDDEN_DIM (new hidden)
    LAST_K = OUT_DIM + HIDDEN_DIM
    
    # Initialize Genes
    GENE1 = np.random.randint(0, MODELLEN, (MODELLEN, 3)) % np.arange(MODELLEN)[:, None]
    GENE2 = np.random.randint(0, len(i0__) + len(i1__) + len(i2__), (MODELLEN, 1))
    
    # Find Active Nodes for the specific input/output counts
    aNodes = activeNode(GENE1, GENE2, N_INPUTS, LAST_K)
    print(f"Active Nodes: {len(aNodes)}")
    
    # Create simple test image (gradient)
    test_img = np.zeros((IMG_SIZE, IMG_SIZE))
    for y in range(IMG_SIZE):
        for x in range(IMG_SIZE):
            test_img[y, x] = (y + x) / (2 * IMG_SIZE)
            
    # Execute Grid RNN CGP
    print("Executing Grid RNN CGP with 3-channel output...")
    start_time = time.time()
    out_img, final_hidden = execGridRNN(GENE1, GENE2, aNodes, test_img, HIDDEN_DIM, OUT_DIM)
    end_time = time.time()
    print(f"Execution finished in {end_time - start_time:.4f} seconds.")
    
    # Normalize output image for visualization (simple clip and scale)
    out_img_sum = np.sum(np.abs(out_img))
    if out_img_sum > 0:
        out_img = (out_img - np.min(out_img)) / (np.max(out_img) - np.min(out_img) + 1e-12)

    # Plot results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Input Image (Gradient)")
    plt.imshow(test_img, cmap='gray')
    
    plt.subplot(1, 3, 2)
    plt.title(f"Output Image ({OUT_DIM} channels)")
    # If OUT_DIM is 3, imshow handles it as RGB. If not 3, imshow might fail or show weirdly.
    # For visualization, we'll try to show it as RGB if it's 3, otherwise just 1st channel.
    if OUT_DIM == 3:
        plt.imshow(out_img)
    else:
        plt.imshow(out_img[:, :, 0], cmap='gray')
        
    plt.subplot(1, 3, 3)
    plt.title("Hidden State (1st channel)")
    plt.imshow(final_hidden[:, :, 0], cmap='viridis')
    
    plt.tight_layout()
    plt.show()