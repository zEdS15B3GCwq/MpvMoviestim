# ❯  python .\bench_array_row_fill_v2.py --cycles 1000 --repeats 3
# cycles=1,000, repeats=3

# 15 values per row
#   local array + slice copy            1.38 ms     91.99 ns/value   1.00x
#   preallocated array + slice copy      1.44 ms     96.03 ns/value   1.04x
#   local preallocated array + slice copy      1.45 ms     96.35 ns/value   1.05x
#   repeated array allocation + slice copy      1.59 ms    106.19 ns/value   1.15x
#   direct target indexed writes        1.67 ms    111.26 ns/value   1.21x
#   scratch list + indexed copy         1.95 ms    129.83 ns/value   1.41x
#   preallocated array + indexed copy      2.45 ms    163.58 ns/value   1.78x

# 20 values per row
#   local array + slice copy            1.23 ms     61.72 ns/value   1.00x
#   local preallocated array + slice copy      1.29 ms     64.61 ns/value   1.05x
#   repeated array allocation + slice copy      1.43 ms     71.29 ns/value   1.15x
#   preallocated array + slice copy      1.43 ms     71.60 ns/value   1.16x
#   direct target indexed writes        1.67 ms     83.59 ns/value   1.35x
#   scratch list + indexed copy         1.95 ms     97.51 ns/value   1.58x
#   preallocated array + indexed copy      2.45 ms    122.73 ns/value   1.99x

# 128 values per row
#   local array + slice copy            1.21 ms      9.48 ns/value   1.00x
#   local preallocated array + slice copy      1.30 ms     10.15 ns/value   1.07x
#   preallocated array + slice copy      1.31 ms     10.22 ns/value   1.08x
#   repeated array allocation + slice copy      1.45 ms     11.33 ns/value   1.19x
#   direct target indexed writes        1.49 ms     11.68 ns/value   1.23x
#   scratch list + indexed copy         1.76 ms     13.72 ns/value   1.45x
#   preallocated array + indexed copy      2.20 ms     17.20 ns/value   1.81x

# 512 values per row
#   local array + slice copy            1.22 ms      2.38 ns/value   1.00x
#   preallocated array + slice copy      1.29 ms      2.53 ns/value   1.06x
#   local preallocated array + slice copy      1.30 ms      2.53 ns/value   1.07x
#   repeated array allocation + slice copy      1.43 ms      2.80 ns/value   1.18x
#   direct target indexed writes        1.51 ms      2.94 ns/value   1.24x
#   scratch list + indexed copy         1.76 ms      3.43 ns/value   1.44x
#   preallocated array + indexed copy      2.18 ms      4.26 ns/value   1.79x
"""Benchmark ways to fill rows in a pre-allocated ``array('d')``.

Run with, for example:
    python bench_array_row_fill.py --cycles 10000 --repeats 5
"""

import argparse
from array import array
from time import perf_counter

FIELD_COUNT = 25


class Fields:
    __slots__ = tuple(f"FIELD_{index}" for index in range(FIELD_COUNT))

    def __init__(self):
        for index, name in enumerate(self.__slots__):
            setattr(self, name, index)


F = Fields()


class BaseProvider:
    def __init__(self, numrows):
        self.numrows = numrows
        self.scratch_array = array("d", [0.0]) * FIELD_COUNT
        self.scratch_list = [0.0] * FIELD_COUNT
        self.value = 0
        self.target = array("d", [0.0]) * (numrows * FIELD_COUNT)

    def base(self, row):
        return row * FIELD_COUNT

    def getvalue(self):
        t = self.value
        self.value += 1
        return t


def fill_direct(provider, numrows):
    for row in range(numrows):
        base = provider.base(row)
        provider.target[base + F.FIELD_0] = provider.getvalue()
        provider.target[base + F.FIELD_1] = provider.getvalue()
        provider.target[base + F.FIELD_2] = provider.getvalue()
        provider.target[base + F.FIELD_3] = provider.getvalue()
        provider.target[base + F.FIELD_4] = provider.getvalue()
        provider.target[base + F.FIELD_5] = provider.getvalue()
        provider.target[base + F.FIELD_6] = provider.getvalue()
        provider.target[base + F.FIELD_7] = provider.getvalue()
        provider.target[base + F.FIELD_8] = provider.getvalue()
        provider.target[base + F.FIELD_9] = provider.getvalue()
        provider.target[base + F.FIELD_10] = provider.getvalue()
        provider.target[base + F.FIELD_11] = provider.getvalue()
        provider.target[base + F.FIELD_12] = provider.getvalue()
        provider.target[base + F.FIELD_13] = provider.getvalue()
        provider.target[base + F.FIELD_14] = provider.getvalue()
        provider.target[base + F.FIELD_15] = provider.getvalue()
        provider.target[base + F.FIELD_16] = provider.getvalue()
        provider.target[base + F.FIELD_17] = provider.getvalue()
        provider.target[base + F.FIELD_18] = provider.getvalue()
        provider.target[base + F.FIELD_19] = provider.getvalue()
        provider.target[base + F.FIELD_20] = provider.getvalue()
        provider.target[base + F.FIELD_21] = provider.getvalue()
        provider.target[base + F.FIELD_22] = provider.getvalue()
        provider.target[base + F.FIELD_23] = provider.getvalue()
        provider.target[base + F.FIELD_24] = provider.getvalue()


def fill_scratch_list_then_index_copy(provider, numrows):
    for row in range(numrows):
        local = provider.scratch_list
        local[F.FIELD_0] = provider.getvalue()
        local[F.FIELD_1] = provider.getvalue()
        local[F.FIELD_2] = provider.getvalue()
        local[F.FIELD_3] = provider.getvalue()
        local[F.FIELD_4] = provider.getvalue()
        local[F.FIELD_5] = provider.getvalue()
        local[F.FIELD_6] = provider.getvalue()
        local[F.FIELD_7] = provider.getvalue()
        local[F.FIELD_8] = provider.getvalue()
        local[F.FIELD_9] = provider.getvalue()
        local[F.FIELD_10] = provider.getvalue()
        local[F.FIELD_11] = provider.getvalue()
        local[F.FIELD_12] = provider.getvalue()
        local[F.FIELD_13] = provider.getvalue()
        local[F.FIELD_14] = provider.getvalue()
        local[F.FIELD_15] = provider.getvalue()
        local[F.FIELD_16] = provider.getvalue()
        local[F.FIELD_17] = provider.getvalue()
        local[F.FIELD_18] = provider.getvalue()
        local[F.FIELD_19] = provider.getvalue()
        local[F.FIELD_20] = provider.getvalue()
        local[F.FIELD_21] = provider.getvalue()
        local[F.FIELD_22] = provider.getvalue()
        local[F.FIELD_23] = provider.getvalue()
        local[F.FIELD_24] = provider.getvalue()
        base = provider.base(row)
        for index in range(FIELD_COUNT):
            provider.target[base + index] = local[index]


def fill_scratch_array_then_index_copy(provider, numrows):
    for row in range(numrows):
        local = provider.scratch_array
        local[F.FIELD_0] = provider.getvalue()
        local[F.FIELD_1] = provider.getvalue()
        local[F.FIELD_2] = provider.getvalue()
        local[F.FIELD_3] = provider.getvalue()
        local[F.FIELD_4] = provider.getvalue()
        local[F.FIELD_5] = provider.getvalue()
        local[F.FIELD_6] = provider.getvalue()
        local[F.FIELD_7] = provider.getvalue()
        local[F.FIELD_8] = provider.getvalue()
        local[F.FIELD_9] = provider.getvalue()
        local[F.FIELD_10] = provider.getvalue()
        local[F.FIELD_11] = provider.getvalue()
        local[F.FIELD_12] = provider.getvalue()
        local[F.FIELD_13] = provider.getvalue()
        local[F.FIELD_14] = provider.getvalue()
        local[F.FIELD_15] = provider.getvalue()
        local[F.FIELD_16] = provider.getvalue()
        local[F.FIELD_17] = provider.getvalue()
        local[F.FIELD_18] = provider.getvalue()
        local[F.FIELD_19] = provider.getvalue()
        local[F.FIELD_20] = provider.getvalue()
        local[F.FIELD_21] = provider.getvalue()
        local[F.FIELD_22] = provider.getvalue()
        local[F.FIELD_23] = provider.getvalue()
        local[F.FIELD_24] = provider.getvalue()
        base = provider.base(row)
        for index in range(FIELD_COUNT):
            provider.target[base + index] = local[index]


def fill_scratch_array_then_slice_copy(provider, numrows):
    for row in range(numrows):
        local = provider.scratch_array
        local[F.FIELD_0] = provider.getvalue()
        local[F.FIELD_1] = provider.getvalue()
        local[F.FIELD_2] = provider.getvalue()
        local[F.FIELD_3] = provider.getvalue()
        local[F.FIELD_4] = provider.getvalue()
        local[F.FIELD_5] = provider.getvalue()
        local[F.FIELD_6] = provider.getvalue()
        local[F.FIELD_7] = provider.getvalue()
        local[F.FIELD_8] = provider.getvalue()
        local[F.FIELD_9] = provider.getvalue()
        local[F.FIELD_10] = provider.getvalue()
        local[F.FIELD_11] = provider.getvalue()
        local[F.FIELD_12] = provider.getvalue()
        local[F.FIELD_13] = provider.getvalue()
        local[F.FIELD_14] = provider.getvalue()
        local[F.FIELD_15] = provider.getvalue()
        local[F.FIELD_16] = provider.getvalue()
        local[F.FIELD_17] = provider.getvalue()
        local[F.FIELD_18] = provider.getvalue()
        local[F.FIELD_19] = provider.getvalue()
        local[F.FIELD_20] = provider.getvalue()
        local[F.FIELD_21] = provider.getvalue()
        local[F.FIELD_22] = provider.getvalue()
        local[F.FIELD_23] = provider.getvalue()
        local[F.FIELD_24] = provider.getvalue()
        base = provider.base(row)
        provider.target[base : base + FIELD_COUNT] = local


def fill_local_array_then_slice_copy(provider, numrows):
    for row in range(numrows):
        FIELD_0 = provider.getvalue()
        FIELD_1 = provider.getvalue()
        FIELD_2 = provider.getvalue()
        FIELD_3 = provider.getvalue()
        FIELD_4 = provider.getvalue()
        FIELD_5 = provider.getvalue()
        FIELD_6 = provider.getvalue()
        FIELD_7 = provider.getvalue()
        FIELD_8 = provider.getvalue()
        FIELD_9 = provider.getvalue()
        FIELD_10 = provider.getvalue()
        FIELD_11 = provider.getvalue()
        FIELD_12 = provider.getvalue()
        FIELD_13 = provider.getvalue()
        FIELD_14 = provider.getvalue()
        FIELD_15 = provider.getvalue()
        FIELD_16 = provider.getvalue()
        FIELD_17 = provider.getvalue()
        FIELD_18 = provider.getvalue()
        FIELD_19 = provider.getvalue()
        FIELD_20 = provider.getvalue()
        FIELD_21 = provider.getvalue()
        FIELD_22 = provider.getvalue()
        FIELD_23 = provider.getvalue()
        FIELD_24 = provider.getvalue()
        base = provider.base(row)
        provider.target[base : base + FIELD_COUNT] = array(
            "d",
            [
                FIELD_0,
                FIELD_1,
                FIELD_2,
                FIELD_3,
                FIELD_4,
                FIELD_5,
                FIELD_6,
                FIELD_7,
                FIELD_8,
                FIELD_9,
                FIELD_10,
                FIELD_11,
                FIELD_12,
                FIELD_13,
                FIELD_14,
                FIELD_15,
                FIELD_16,
                FIELD_17,
                FIELD_18,
                FIELD_19,
                FIELD_20,
                FIELD_21,
                FIELD_22,
                FIELD_23,
                FIELD_24,
            ],
        )


def fill_locally_allocated_array_then_slice_copy(provider, numrows):
    local = array("d", [0.0]) * FIELD_COUNT
    for row in range(numrows):
        local = provider.scratch_array
        local[F.FIELD_0] = provider.getvalue()
        local[F.FIELD_1] = provider.getvalue()
        local[F.FIELD_2] = provider.getvalue()
        local[F.FIELD_3] = provider.getvalue()
        local[F.FIELD_4] = provider.getvalue()
        local[F.FIELD_5] = provider.getvalue()
        local[F.FIELD_6] = provider.getvalue()
        local[F.FIELD_7] = provider.getvalue()
        local[F.FIELD_8] = provider.getvalue()
        local[F.FIELD_9] = provider.getvalue()
        local[F.FIELD_10] = provider.getvalue()
        local[F.FIELD_11] = provider.getvalue()
        local[F.FIELD_12] = provider.getvalue()
        local[F.FIELD_13] = provider.getvalue()
        local[F.FIELD_14] = provider.getvalue()
        local[F.FIELD_15] = provider.getvalue()
        local[F.FIELD_16] = provider.getvalue()
        local[F.FIELD_17] = provider.getvalue()
        local[F.FIELD_18] = provider.getvalue()
        local[F.FIELD_19] = provider.getvalue()
        local[F.FIELD_20] = provider.getvalue()
        local[F.FIELD_21] = provider.getvalue()
        local[F.FIELD_22] = provider.getvalue()
        local[F.FIELD_23] = provider.getvalue()
        local[F.FIELD_24] = provider.getvalue()
        base = provider.base(row)
        provider.target[base : base + FIELD_COUNT] = local


def fill_repeatedly_allocated_array_then_slice_copy(provider, numrows):
    for row in range(numrows):
        local = array("d", [0.0]) * FIELD_COUNT
        local = provider.scratch_array
        local[F.FIELD_0] = provider.getvalue()
        local[F.FIELD_1] = provider.getvalue()
        local[F.FIELD_2] = provider.getvalue()
        local[F.FIELD_3] = provider.getvalue()
        local[F.FIELD_4] = provider.getvalue()
        local[F.FIELD_5] = provider.getvalue()
        local[F.FIELD_6] = provider.getvalue()
        local[F.FIELD_7] = provider.getvalue()
        local[F.FIELD_8] = provider.getvalue()
        local[F.FIELD_9] = provider.getvalue()
        local[F.FIELD_10] = provider.getvalue()
        local[F.FIELD_11] = provider.getvalue()
        local[F.FIELD_12] = provider.getvalue()
        local[F.FIELD_13] = provider.getvalue()
        local[F.FIELD_14] = provider.getvalue()
        local[F.FIELD_15] = provider.getvalue()
        local[F.FIELD_16] = provider.getvalue()
        local[F.FIELD_17] = provider.getvalue()
        local[F.FIELD_18] = provider.getvalue()
        local[F.FIELD_19] = provider.getvalue()
        local[F.FIELD_20] = provider.getvalue()
        local[F.FIELD_21] = provider.getvalue()
        local[F.FIELD_22] = provider.getvalue()
        local[F.FIELD_23] = provider.getvalue()
        local[F.FIELD_24] = provider.getvalue()
        base = provider.base(row)
        provider.target[base : base + FIELD_COUNT] = local


METHODS = (
    ("direct target indexed writes", fill_direct),
    ("scratch list + indexed copy", fill_scratch_list_then_index_copy),
    ("preallocated array + indexed copy", fill_scratch_array_then_index_copy),
    ("preallocated array + slice copy", fill_scratch_array_then_slice_copy),
    ("local array + slice copy", fill_local_array_then_slice_copy),
    (
        "repeated array allocation + slice copy",
        fill_repeatedly_allocated_array_then_slice_copy,
    ),
    (
        "local preallocated array + slice copy",
        fill_locally_allocated_array_then_slice_copy,
    ),
)


def check(method):
    provider = BaseProvider(1)
    method(provider, 1)
    assert provider.target == array("d", range(FIELD_COUNT))


def benchmark(method, numrows):
    provider = BaseProvider(numrows)
    began = perf_counter()
    method(provider, numrows)
    return perf_counter() - began


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    sizes = (15, 20, 128, 512)

    print(f"cycles={args.cycles:,}, repeats={args.repeats}")
    for n_slots in sizes:
        for _, method in METHODS:
            check(method)
        results = []
        for name, method in METHODS:
            samples = [benchmark(method, args.cycles) for _ in range(args.repeats)]
            results.append((min(samples), name))
        fastest = min(value for value, _ in results)
        print(f"\n{n_slots} values per row")
        for elapsed, name in sorted(results):
            print(
                f"  {name:30} {elapsed * 1e3:9.2f} ms  {elapsed / args.cycles / n_slots * 1e9:8.2f} ns/value  {elapsed / fastest:5.2f}x"
            )


if __name__ == "__main__":
    main()
