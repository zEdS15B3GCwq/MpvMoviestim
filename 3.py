import numpy as np
from array import array
from time import perf_counter

# -------------------------
# Benchmark parameters
# -------------------------
N = 5_000_000   # number of rows
K = 20          # timings per row


# -------------------------
# Measure perf_counter() overhead
# -------------------------
t0 = perf_counter()
for _ in range(N * K):
    perf_counter()
perf_overhead = perf_counter() - t0
print(f"perf_counter() overhead: {perf_overhead}, which equals {perf_overhead / N / K} per iteration")


# -------------------------
# NumPy benchmark
# -------------------------
a = np.empty((N, K), dtype=np.float64)
row_np = np.empty(K, dtype=np.float64)

t0 = perf_counter()
for i in range(N):
    for j in range(K):
        row_np[j] = perf_counter()
    a[i] = row_np
t_numpy = perf_counter() - t0 - perf_overhead
print(f"numpy: {t_numpy}, {t_numpy / K / N}")


# -------------------------
# array('d') benchmark
# -------------------------
b = array('d', [0.0]) * (N * K)

t0 = perf_counter()
for i in range(N):
    base = i * K
    for j in range(K):
        b[base + j] = perf_counter()
t_array = perf_counter() - t0 - perf_overhead
print(f"array: {t_array}, {t_array / K / N}")

