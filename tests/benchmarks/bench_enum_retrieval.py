"""Benchmark numeric field lookup strategies.

Run with, for example:
    python bench_enum_retrieval.py --cycles 10000 --repeats 5
"""

import argparse
import time
from enum import Enum, IntEnum


class IntFields(IntEnum):
    pass


class EnumFields(Enum):
    pass


FIELD_COUNT = 1024
IntFields = IntEnum(
    "IntFields", {f"FIELD_{index}": index for index in range(FIELD_COUNT)}
)
EnumFields = Enum(
    "EnumFields", {f"FIELD_{index}": index for index in range(FIELD_COUNT)}
)


class PlainFields:
    pass


for _index in range(FIELD_COUNT):
    setattr(PlainFields, f"FIELD_{_index}", _index)


class SlottedFields:
    __slots__ = tuple(f"field_{index}" for index in range(FIELD_COUNT))

    def __init__(self):
        for index in range(FIELD_COUNT):
            setattr(self, f"field_{index}", index)


SLOTTED_FIELDS = SlottedFields()
PLAIN_NAMES = tuple(f"FIELD_{index}" for index in range(FIELD_COUNT))
SLOTTED_NAMES = tuple(f"field_{index}" for index in range(FIELD_COUNT))
FIELD_TUPLE = tuple(range(FIELD_COUNT))
FIELD_LIST = list(range(FIELD_COUNT))
FIELD_DICT = {index: index for index in range(FIELD_COUNT)}


def retrieve_int_enum(start, n_slots):
    total = 0
    for offset in range(n_slots):
        total += IntFields(start + offset).value
    return total


def retrieve_enum(start, n_slots):
    total = 0
    for offset in range(n_slots):
        total += EnumFields(start + offset).value
    return total


def retrieve_plain_class(start, n_slots):
    total = 0
    for offset in range(n_slots):
        total += getattr(PlainFields, PLAIN_NAMES[start + offset])
    return total


def retrieve_slots(start, n_slots):
    total = 0
    for offset in range(n_slots):
        total += getattr(SLOTTED_FIELDS, SLOTTED_NAMES[start + offset])
    return total


def retrieve_tuple(start, n_slots):
    total = 0
    for offset in range(n_slots):
        total += FIELD_TUPLE[start + offset]
    return total


def retrieve_list(start, n_slots):
    total = 0
    for offset in range(n_slots):
        total += FIELD_LIST[start + offset]
    return total


def retrieve_dict(start, n_slots):
    total = 0
    for offset in range(n_slots):
        total += FIELD_DICT[start + offset]
    return total


METHODS = (
    ("IntEnum(value).value", retrieve_int_enum),
    ("Enum(value).value", retrieve_enum),
    ("plain class + getattr", retrieve_plain_class),
    ("slotted instance + getattr", retrieve_slots),
    ("tuple[index]", retrieve_tuple),
    ("list[index]", retrieve_list),
    ("dict[index]", retrieve_dict),
)


def run_check(cycles, n_slots):
    expected = n_slots * (n_slots - 1) // 2
    for _, method in METHODS:
        assert (
            sum(
                method(start, n_slots)
                for start in range(cycles % (FIELD_COUNT - n_slots))
            )
            >= 0
        )
        assert method(0, n_slots) == expected


def benchmark(method, cycles, n_slots):
    starts = tuple(index % (FIELD_COUNT - n_slots + 1) for index in range(cycles))
    began = time.perf_counter()
    total = 0
    for start in starts:
        total += method(start, n_slots)
    elapsed = time.perf_counter() - began
    return elapsed, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    sizes = (15, 20, 128, 512)

    print(f"cycles={args.cycles:,}, repeats={args.repeats}, field_count={FIELD_COUNT}")
    for n_slots in sizes:
        run_check(args.cycles, n_slots)
        results = []
        for name, method in METHODS:
            samples = [
                benchmark(method, args.cycles, n_slots)[0] for _ in range(args.repeats)
            ]
            results.append((min(samples), name))
        fastest = min(value for value, _ in results)
        print(f"\n{n_slots} sequential retrievals")
        for elapsed, name in sorted(results):
            print(
                f"  {name:28} {elapsed * 1e3:9.2f} ms  {elapsed / args.cycles / n_slots * 1e9:8.2f} ns/retrieval  {elapsed / fastest:5.2f}x"
            )


if __name__ == "__main__":
    main()
