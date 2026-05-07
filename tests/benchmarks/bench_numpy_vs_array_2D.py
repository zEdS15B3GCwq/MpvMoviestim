import numpy as np
from array import array
from time import perf_counter

N = 500
K = 20

# -------------------------
# NumPy (fastest method)
# -------------------------
a = np.empty((N, K), dtype=np.float64)
row_np = np.empty(K, dtype=np.float64)

t0 = perf_counter()
for _ in range(1000):
    for i in range(N):
        for j in range(K):
            row_np[j] = perf_counter()
        a[i] = row_np
t_numpy = perf_counter() - t0
print("numpy:", t_numpy)


# -------------------------
# array('d') (fastest possible method)
# -------------------------
b = array('d', [0.0]) * (N * K)

t0 = perf_counter()
for _ in range(1000):
    for i in range(N):
        base = i * K
        for j in range(K):
            b[base + j] = perf_counter()
t_array = perf_counter() - t0
print("array:", t_array)
