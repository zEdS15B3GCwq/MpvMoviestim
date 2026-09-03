# ❯  python .\bench_array_row_fill_v2.py --cycles 1000 --repeats 3
# cycles=1,000, repeats=3

# 15 values per row
#   local array + slice copy            1.35 ms     89.75 ns/value   1.00x
#   preallocated array + slice copy      1.48 ms     98.95 ns/value   1.10x
#   local preallocated array + slice copy      1.48 ms     98.97 ns/value   1.10x
#   repeated array allocation + slice copy      1.64 ms    109.05 ns/value   1.21x
#   direct target indexed writes        1.69 ms    112.45 ns/value   1.25x
#   scratch list + indexed copy         1.98 ms    131.93 ns/value   1.47x
#   preallocated array + indexed copy      2.50 ms    166.35 ns/value   1.85x

# 20 values per row
#   local array + slice copy            1.26 ms     63.15 ns/value   1.00x
#   local preallocated array + slice copy      1.33 ms     66.70 ns/value   1.06x
#   repeated array allocation + slice copy      1.47 ms     73.34 ns/value   1.16x
#   preallocated array + slice copy      1.49 ms     74.27 ns/value   1.18x
#   direct target indexed writes        1.68 ms     84.22 ns/value   1.33x
#   scratch list + indexed copy         1.93 ms     96.33 ns/value   1.53x
#   preallocated array + indexed copy      2.48 ms    124.16 ns/value   1.97x

# 128 values per row
#   local array + slice copy            1.20 ms      9.40 ns/value   1.00x
#   local preallocated array + slice copy      1.33 ms     10.40 ns/value   1.11x
#   preallocated array + slice copy      1.34 ms     10.43 ns/value   1.11x
#   repeated array allocation + slice copy      1.47 ms     11.45 ns/value   1.22x
#   direct target indexed writes        1.51 ms     11.80 ns/value   1.26x
#   scratch list + indexed copy         1.75 ms     13.66 ns/value   1.45x
#   preallocated array + indexed copy      2.25 ms     17.59 ns/value   1.87x

# 512 values per row
#   local array + slice copy            1.22 ms      2.39 ns/value   1.00x
#   preallocated array + slice copy      1.33 ms      2.60 ns/value   1.09x
#   local preallocated array + slice copy      1.33 ms      2.60 ns/value   1.09x
#   repeated array allocation + slice copy      1.46 ms      2.85 ns/value   1.20x
#   direct target indexed writes        1.51 ms      2.94 ns/value   1.23x
#   scratch list + indexed copy         1.74 ms      3.39 ns/value   1.42x
#   preallocated array + indexed copy      2.22 ms      4.33 ns/value   1.82x
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
        for index in range(FIELD_COUNT):
            setattr(self, f"FIELD_{index}", index)


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
