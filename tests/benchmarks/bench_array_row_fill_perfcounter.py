"""Benchmark ways to fill rows in a pre-allocated ``array('d')``.

Run with, for example:
    python bench_array_row_fill.py --cycles 10000 --repeats 5
"""

import argparse
import time
from array import array


class BaseProvider:
    def __init__(self, n_slots):
        self.n_slots = n_slots
        self.scratch_array = array("d", [0.0]) * n_slots
        self.scratch_list = [0.0] * n_slots

    def base(self, row):
        return row * self.n_slots


VALUES = tuple(index + 0.25 for index in range(1024))


def fill_direct(target, provider, row, n_slots):
    base = provider.base(row)
    for index in range(n_slots):
        target[base + index] = VALUES[index]


def fill_local_list_then_index_copy(target, provider, row, n_slots):
    base = provider.base(row)
    local = provider.scratch_list
    for index in range(n_slots):
        local[index] = VALUES[index]
    for index in range(n_slots):
        target[base + index] = local[index]


def fill_local_array_then_index_copy(target, provider, row, n_slots):
    base = provider.base(row)
    local = provider.scratch_array
    for index in range(n_slots):
        local[index] = VALUES[index]
    for index in range(n_slots):
        target[base + index] = local[index]


def fill_local_array_then_slice_copy(target, provider, row, n_slots):
    base = provider.base(row)
    local = provider.scratch_array
    for index in range(n_slots):
        local[index] = VALUES[index]
    target[base : base + n_slots] = local


def fill_array_fromlist_then_slice_copy(target, provider, row, n_slots):
    base = provider.base(row)
    local = array("d", VALUES[:n_slots])
    target[base : base + n_slots] = local


def fill_array_fromlist_each_cycle(target, provider, row, n_slots):
    base = provider.base(row)
    target[base : base + n_slots] = array("d", VALUES[:n_slots])


METHODS = (
    ("direct target indexed writes", fill_direct),
    ("local list + indexed copy", fill_local_list_then_index_copy),
    ("local array + indexed copy", fill_local_array_then_index_copy),
    ("local array + slice copy", fill_local_array_then_slice_copy),
    ("array.fromlist + slice copy", fill_array_fromlist_then_slice_copy),
    ("array allocation + slice copy", fill_array_fromlist_each_cycle),
)


def make_target(capacity, n_slots):
    return array("d", [0.0]) * (capacity * n_slots)


def check(method, n_slots):
    provider = BaseProvider(n_slots)
    target = make_target(2, n_slots)
    method(target, provider, 1, n_slots)
    assert target[n_slots : 2 * n_slots] == array("d", VALUES[:n_slots])
    assert all(value == 0.0 for value in target[:n_slots])


def benchmark(method, cycles, n_slots):
    provider = BaseProvider(n_slots)
    target = make_target(cycles, n_slots)
    began = time.perf_counter()
    for row in range(cycles):
        method(target, provider, row, n_slots)
    return time.perf_counter() - began, target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    sizes = (15, 20, 128, 512)

    print(f"cycles={args.cycles:,}, repeats={args.repeats}")
    for n_slots in sizes:
        for _, method in METHODS:
            check(method, n_slots)
        results = []
        for name, method in METHODS:
            samples = [
                benchmark(method, args.cycles, n_slots)[0] for _ in range(args.repeats)
            ]
            results.append((min(samples), name))
        fastest = min(value for value, _ in results)
        print(f"\n{n_slots} values per row")
        for elapsed, name in sorted(results):
            print(
                f"  {name:30} {elapsed * 1e3:9.2f} ms  {elapsed / args.cycles / n_slots * 1e9:8.2f} ns/value  {elapsed / fastest:5.2f}x"
            )


if __name__ == "__main__":
    main()
