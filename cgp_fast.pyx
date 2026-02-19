import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, tanh, sqrt, fabs, log, pow, M_PI
import cython

# Use float32 for consistency with the rest of the code
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float exec_node_c(int func_id, float x, float y, float z) nogil:
    # i0__ functions
    if func_id == 0: return sin(x)
    if func_id == 1: return cos(x)
    if func_id == 2: return sin(x * M_PI)
    if func_id == 3: return cos(x * M_PI)
    if func_id == 4: return x * 2.0
    if func_id == 5: return x * 10.0
    if func_id == 6: return x * 0.1
    if func_id == 7: return x * 0.5
    if func_id == 8: return x * 0.9
    if func_id == 9: return x + 1.0
    if func_id == 10: return x - 1.0
    if func_id == 11: return x + 10.0
    if func_id == 12: return x - 10.0
    if func_id == 13: return -x
    if func_id == 14: return x / 2.0
    if func_id == 15: return tanh(x)
    if func_id == 16: return x * x
    if func_id == 17: return fabs(x)
    if func_id == 18: return sqrt(fabs(x))
    if func_id == 19: return pow(fabs(x), 1.0/3.0) * (1.0 if x >= 0 else -1.0)
    if func_id == 20: return log(x*x + 1e-12)
    if func_id == 21: return x
    if func_id == 22: return x if x > 0 else 0
    if func_id == 23: return x if x < 0 else 0
    if func_id == 24: return x + sin(x) * sin(x)
    
    # i1__ functions (shifted by 25)
    cdef int f1 = func_id - 25
    if f1 == 0: return x + y
    if f1 == 1: return x - y
    if f1 == 2: return x * y
    if f1 == 3: return x + y * 0.5
    if f1 == 4: return x + y * 0.1
    if f1 == 5: return x + y * 0.9
    if f1 == 6: return x / (y * y + 1e-12)
    if f1 == 7: return x if x > y else y
    if f1 == 8: return x if x < y else y
    if f1 == 9: return sqrt(x*x + y*y)
    if f1 == 10: return (x + y) / 2.0
    if f1 == 11: return (x - y) * (x - y)
    if f1 == 12: return x * tanh(y)
    if f1 == 13: return x * (tanh(y) + 1.0)
    if f1 == 14: return sin(x * M_PI * y)
    if f1 == 15: return cos(x * M_PI * y)
    
    # i2__ functions (shifted by 25 + 16 = 41)
    cdef int f2 = func_id - 41
    if f2 == 0: return x + y + z
    if f2 == 1: return x + y - z
    if f2 == 2: return x - y - z
    if f2 == 3: return x * y * z
    if f2 == 4: return x * (y*y + 1e-12) / (z*z + 1e-12)
    if f2 == 5: return x / (y*y + z*z + 1e-12)
    cdef float m1 = x if x > y else y
    if f2 == 6: return m1 if m1 > z else z
    cdef float mi1 = x if x < y else y
    if f2 == 7: return mi1 if mi1 < z else z
    if f2 == 8: return x if x > (y if y < z else z) else (y if y < z else z)
    if f2 == 9: return sqrt(x*x + y*y + z*z)
    if f2 == 10: return sqrt((x-y)*(x-y) + (y-z)*(y-z) + (z-x)*(z-x))
    if f2 == 11: 
        m1 = x * y * z
        return pow(fabs(m1), 1.0/3.0) * (1.0 if m1 >= 0 else -1.0)
    if f2 == 12:
        m1 = (x-y) * (y-z) * (z-x)
        return pow(fabs(m1), 1.0/3.0) * (1.0 if m1 >= 0 else -1.0)
    if f2 == 13: return (x + y + z) / 3.0
    if f2 == 14: return sin(x * M_PI) * (1.0 + tanh(y)) + cos(z * M_PI) * (1.0 - tanh(x))
    
    return 0.0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def execGridRNN_cython(int[:,:] gene1, int[:,:] gene2, int[:] active_nodes, float[:,:,:] image, int hidden_dim, int out_dim):
    cdef int H = image.shape[0]
    cdef int W = image.shape[1]
    cdef int C = image.shape[2]
    cdef int n_nodes = gene1.shape[0] # MODELLEN
    cdef int last_k = out_dim + hidden_dim
    cdef int n_inputs = C + 2 * hidden_dim
    cdef int total_calc = n_inputs + n_nodes
    cdef int n_active = active_nodes.shape[0]
    
    cdef np.ndarray[DTYPE_t, ndim=3] hidden = np.zeros((H + 1, W + 1, hidden_dim), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] output = np.zeros((H, W, out_dim), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] calculated = np.zeros(total_calc, dtype=DTYPE)
    
    cdef int y, x, c, i, n, sid, fid, local_n
    cdef float val_x, val_y, val_z
    
    for y in range(H):
        for x in range(W):
            # Set Inputs
            for c in range(C):
                calculated[c] = image[y, x, c]
            for i in range(hidden_dim):
                calculated[C + i] = hidden[y, x + 1, i] # top
                calculated[C + hidden_dim + i] = hidden[y + 1, x, i] # left
                
            # Exec Active Nodes
            for i in range(n_active):
                n = active_nodes[i]
                if n < n_inputs:
                    continue
                
                local_n = n - n_inputs
                fid = gene2[local_n, 0]
                val_x = calculated[gene1[local_n, 0]]
                val_y = calculated[gene1[local_n, 1]]
                val_z = calculated[gene1[local_n, 2]]
                
                calculated[n] = exec_node_c(fid, val_x, val_y, val_z)
                
            # Store Results
            for i in range(out_dim):
                output[y, x, i] = calculated[total_calc - last_k + i]
            for i in range(hidden_dim):
                hidden[y + 1, x + 1, i] = calculated[total_calc - last_k + out_dim + i]
                
    return output
