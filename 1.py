import numpy as np
from time import perf_counter
from array import array

n = 1000000  # 1 million writes

# measure perf_counter() overhead
t0 = perf_counter()
for _ in range(n):
    perf_counter()
t_perf = perf_counter() - t0

print("perf_counter() baseline:", t_perf)

# NumPy
a = np.empty(n, dtype=np.float64)
t0 = perf_counter()
for i in range(n):
    a[i] = perf_counter()
t_numpy = perf_counter() - t0 - t_perf
print("numpy:", t_numpy)

# array('d')
b = array('d', [0.0]) * n
v = memoryview(b)
t0 = perf_counter()
for i in range(n):
    v[i] = perf_counter()
t_array = perf_counter() - t0 - t_perf
print("array:", t_array)
