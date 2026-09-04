# ❯  python .\bench_enum_retrieval.py --cycles 1000 --repeats 3
# cycles=1,000, repeats=3, field_count=128

# 15 sequential retrievals
#   slotted instance + direct names      0.16 ms     10.93 ns/retrieval   1.00x
#   classvars + direct names          0.17 ms     11.59 ns/retrieval   1.06x
#   plain class + direct names        0.17 ms     11.59 ns/retrieval   1.06x
#   namedtuple + direct names         0.22 ms     14.75 ns/retrieval   1.35x
#   IntEnum + direct names            0.24 ms     16.07 ns/retrieval   1.47x
#   Dict + direct names               0.26 ms     17.23 ns/retrieval   1.58x
#   named dict[name]                  0.49 ms     32.49 ns/retrieval   2.97x
#   plain class + getattr             0.59 ms     39.25 ns/retrieval   3.59x
#   named tuple + getattr             0.60 ms     40.19 ns/retrieval   3.68x
#   slotted instance + getattr        0.61 ms     40.65 ns/retrieval   3.72x
#   Enum + direct names               1.13 ms     75.39 ns/retrieval   6.90x
#   Enum[name].value                  1.70 ms    113.66 ns/retrieval  10.40x
#   IntEnum[name].value               1.83 ms    122.30 ns/retrieval  11.19x

# 20 sequential retrievals
#   slotted instance + direct names      0.20 ms      9.79 ns/retrieval   1.00x
#   classvars + direct names          0.21 ms     10.59 ns/retrieval   1.08x
#   plain class + direct names        0.21 ms     10.60 ns/retrieval   1.08x
#   namedtuple + direct names         0.29 ms     14.55 ns/retrieval   1.49x
#   IntEnum + direct names            0.31 ms     15.66 ns/retrieval   1.60x
#   Dict + direct names               0.33 ms     16.57 ns/retrieval   1.69x
#   named dict[name]                  0.62 ms     30.99 ns/retrieval   3.17x
#   plain class + getattr             0.73 ms     36.58 ns/retrieval   3.74x
#   named tuple + getattr             0.75 ms     37.59 ns/retrieval   3.84x
#   slotted instance + getattr        0.79 ms     39.30 ns/retrieval   4.01x
#   Enum + direct names               1.50 ms     74.81 ns/retrieval   7.64x
#   Enum[name].value                  2.18 ms    109.23 ns/retrieval  11.16x
#   IntEnum[name].value               2.40 ms    120.06 ns/retrieval  12.26x

# 128 sequential retrievals
#   slotted instance + direct names      1.75 ms     13.64 ns/retrieval   1.00x
#   plain class + direct names        1.85 ms     14.43 ns/retrieval   1.06x
#   classvars + direct names          1.88 ms     14.68 ns/retrieval   1.08x
#   namedtuple + direct names         2.44 ms     19.04 ns/retrieval   1.40x
#   Dict + direct names               2.88 ms     22.48 ns/retrieval   1.65x
#   IntEnum + direct names            2.93 ms     22.90 ns/retrieval   1.68x
#   named dict[name]                  3.85 ms     30.10 ns/retrieval   2.21x
#   plain class + getattr             4.05 ms     31.67 ns/retrieval   2.32x
#   named tuple + getattr             4.15 ms     32.42 ns/retrieval   2.38x
#   slotted instance + getattr        4.33 ms     33.85 ns/retrieval   2.48x
#   Enum + direct names              12.53 ms     97.88 ns/retrieval   7.18x
#   Enum[name].value                 12.94 ms    101.08 ns/retrieval   7.41x
#   IntEnum[name].value              13.83 ms    108.01 ns/retrieval   7.92x
"""Benchmark numeric field lookup strategies.

Run with, for example:
    python bench_enum_retrieval.py --cycles 10000 --repeats 5
"""

import argparse
import time
from collections import namedtuple
from enum import Enum, IntEnum

FIELD_COUNT = 128
IntFields = IntEnum(
    "IntFields", {f"FIELD_{index}": index for index in range(FIELD_COUNT)}
)
EnumFields = Enum(
    "EnumFields", {f"FIELD_{index}": index for index in range(FIELD_COUNT)}
)
NamedTupleFields = namedtuple(
    "NamedTupleFields",
    tuple(f"FIELD_{index}" for index in range(FIELD_COUNT)),
    defaults=range(FIELD_COUNT),
)


class CV:
    FIELD_0 = 0
    FIELD_1 = 1
    FIELD_2 = 2
    FIELD_3 = 3
    FIELD_4 = 4
    FIELD_5 = 5
    FIELD_6 = 6
    FIELD_7 = 7
    FIELD_8 = 8
    FIELD_9 = 9
    FIELD_10 = 10
    FIELD_11 = 11
    FIELD_12 = 12
    FIELD_13 = 13
    FIELD_14 = 14
    FIELD_15 = 15
    FIELD_16 = 16
    FIELD_17 = 17
    FIELD_18 = 18
    FIELD_19 = 19
    FIELD_20 = 20
    FIELD_21 = 21
    FIELD_22 = 22
    FIELD_23 = 23
    FIELD_24 = 24
    FIELD_25 = 25
    FIELD_26 = 26
    FIELD_27 = 27
    FIELD_28 = 28
    FIELD_29 = 29
    FIELD_30 = 30
    FIELD_31 = 31
    FIELD_32 = 32
    FIELD_33 = 33
    FIELD_34 = 34
    FIELD_35 = 35
    FIELD_36 = 36
    FIELD_37 = 37
    FIELD_38 = 38
    FIELD_39 = 39
    FIELD_40 = 40
    FIELD_41 = 41
    FIELD_42 = 42
    FIELD_43 = 43
    FIELD_44 = 44
    FIELD_45 = 45
    FIELD_46 = 46
    FIELD_47 = 47
    FIELD_48 = 48
    FIELD_49 = 49
    FIELD_50 = 50
    FIELD_51 = 51
    FIELD_52 = 52
    FIELD_53 = 53
    FIELD_54 = 54
    FIELD_55 = 55
    FIELD_56 = 56
    FIELD_57 = 57
    FIELD_58 = 58
    FIELD_59 = 59
    FIELD_60 = 60
    FIELD_61 = 61
    FIELD_62 = 62
    FIELD_63 = 63
    FIELD_64 = 64
    FIELD_65 = 65
    FIELD_66 = 66
    FIELD_67 = 67
    FIELD_68 = 68
    FIELD_69 = 69
    FIELD_70 = 70
    FIELD_71 = 71
    FIELD_72 = 72
    FIELD_73 = 73
    FIELD_74 = 74
    FIELD_75 = 75
    FIELD_76 = 76
    FIELD_77 = 77
    FIELD_78 = 78
    FIELD_79 = 79
    FIELD_80 = 80
    FIELD_81 = 81
    FIELD_82 = 82
    FIELD_83 = 83
    FIELD_84 = 84
    FIELD_85 = 85
    FIELD_86 = 86
    FIELD_87 = 87
    FIELD_88 = 88
    FIELD_89 = 89
    FIELD_90 = 90
    FIELD_91 = 91
    FIELD_92 = 92
    FIELD_93 = 93
    FIELD_94 = 94
    FIELD_95 = 95
    FIELD_96 = 96
    FIELD_97 = 97
    FIELD_98 = 98
    FIELD_99 = 99
    FIELD_100 = 100
    FIELD_101 = 101
    FIELD_102 = 102
    FIELD_103 = 103
    FIELD_104 = 104
    FIELD_105 = 105
    FIELD_106 = 106
    FIELD_107 = 107
    FIELD_108 = 108
    FIELD_109 = 109
    FIELD_110 = 110
    FIELD_111 = 111
    FIELD_112 = 112
    FIELD_113 = 113
    FIELD_114 = 114
    FIELD_115 = 115
    FIELD_116 = 116
    FIELD_117 = 117
    FIELD_118 = 118
    FIELD_119 = 119
    FIELD_120 = 120
    FIELD_121 = 121
    FIELD_122 = 122
    FIELD_123 = 123
    FIELD_124 = 124
    FIELD_125 = 125
    FIELD_126 = 126
    FIELD_127 = 127


class PlainClass:
    pass


for _index in range(FIELD_COUNT):
    setattr(PlainClass, f"FIELD_{_index}", _index)


class SlottedClass:
    __slots__ = tuple(f"FIELD_{index}" for index in range(FIELD_COUNT))

    def __init__(self):
        for index, name in enumerate(self.__slots__):
            setattr(self, name, index)


SLOTTED_CLASS = SlottedClass()
PLAINCLASS_NAMES = tuple(f"FIELD_{index}" for index in range(FIELD_COUNT))
NAMED_DICT = {name: index for index, name in enumerate(PLAINCLASS_NAMES)}
NAMED_TUPLE = NamedTupleFields()


def retrieve_int_enum(start, n_slots):
    total = 0
    for offset in range(n_slots):
        total += IntFields[PLAINCLASS_NAMES[start + offset]].value
    return total


def retrieve_enum(start, n_slots):
    total = 0
    for offset in range(n_slots):
        total += EnumFields[PLAINCLASS_NAMES[start + offset]].value
    return total


def retrieve_plain_class(start, n_slots):
    total = 0
    for offset in range(n_slots):
        total += getattr(PlainClass, PLAINCLASS_NAMES[start + offset])
    return total


def retrieve_slots(start, n_slots):
    total = 0
    for offset in range(n_slots):
        total += getattr(SLOTTED_CLASS, PLAINCLASS_NAMES[start + offset])
    return total


def retrieve_named_dict(start, n_slots):
    total = 0
    for offset in range(n_slots):
        total += NAMED_DICT[PLAINCLASS_NAMES[start + offset]]
    return total


def retrieve_named_tuple(start, n_slots):
    total = 0
    for offset in range(n_slots):
        total += getattr(NAMED_TUPLE, PLAINCLASS_NAMES[start + offset])
    return total


DIRECT_SIZES = (15, 20, 128)


def retrieve_slots_direct_15(_start, _n_slots):
    return (
        SLOTTED_CLASS.FIELD_0
        + SLOTTED_CLASS.FIELD_1
        + SLOTTED_CLASS.FIELD_2
        + SLOTTED_CLASS.FIELD_3
        + SLOTTED_CLASS.FIELD_4
        + SLOTTED_CLASS.FIELD_5
        + SLOTTED_CLASS.FIELD_6
        + SLOTTED_CLASS.FIELD_7
        + SLOTTED_CLASS.FIELD_8
        + SLOTTED_CLASS.FIELD_9
        + SLOTTED_CLASS.FIELD_10
        + SLOTTED_CLASS.FIELD_11
        + SLOTTED_CLASS.FIELD_12
        + SLOTTED_CLASS.FIELD_13
        + SLOTTED_CLASS.FIELD_14
    )


def retrieve_slots_direct_20(_start, _n_slots):
    return (
        SLOTTED_CLASS.FIELD_0
        + SLOTTED_CLASS.FIELD_1
        + SLOTTED_CLASS.FIELD_2
        + SLOTTED_CLASS.FIELD_3
        + SLOTTED_CLASS.FIELD_4
        + SLOTTED_CLASS.FIELD_5
        + SLOTTED_CLASS.FIELD_6
        + SLOTTED_CLASS.FIELD_7
        + SLOTTED_CLASS.FIELD_8
        + SLOTTED_CLASS.FIELD_9
        + SLOTTED_CLASS.FIELD_10
        + SLOTTED_CLASS.FIELD_11
        + SLOTTED_CLASS.FIELD_12
        + SLOTTED_CLASS.FIELD_13
        + SLOTTED_CLASS.FIELD_14
        + SLOTTED_CLASS.FIELD_15
        + SLOTTED_CLASS.FIELD_16
        + SLOTTED_CLASS.FIELD_17
        + SLOTTED_CLASS.FIELD_18
        + SLOTTED_CLASS.FIELD_19
    )


def retrieve_slots_direct_128(_start, _n_slots):
    return sum(
        (
            SLOTTED_CLASS.FIELD_0,
            SLOTTED_CLASS.FIELD_1,
            SLOTTED_CLASS.FIELD_2,
            SLOTTED_CLASS.FIELD_3,
            SLOTTED_CLASS.FIELD_4,
            SLOTTED_CLASS.FIELD_5,
            SLOTTED_CLASS.FIELD_6,
            SLOTTED_CLASS.FIELD_7,
            SLOTTED_CLASS.FIELD_8,
            SLOTTED_CLASS.FIELD_9,
            SLOTTED_CLASS.FIELD_10,
            SLOTTED_CLASS.FIELD_11,
            SLOTTED_CLASS.FIELD_12,
            SLOTTED_CLASS.FIELD_13,
            SLOTTED_CLASS.FIELD_14,
            SLOTTED_CLASS.FIELD_15,
            SLOTTED_CLASS.FIELD_16,
            SLOTTED_CLASS.FIELD_17,
            SLOTTED_CLASS.FIELD_18,
            SLOTTED_CLASS.FIELD_19,
            SLOTTED_CLASS.FIELD_20,
            SLOTTED_CLASS.FIELD_21,
            SLOTTED_CLASS.FIELD_22,
            SLOTTED_CLASS.FIELD_23,
            SLOTTED_CLASS.FIELD_24,
            SLOTTED_CLASS.FIELD_25,
            SLOTTED_CLASS.FIELD_26,
            SLOTTED_CLASS.FIELD_27,
            SLOTTED_CLASS.FIELD_28,
            SLOTTED_CLASS.FIELD_29,
            SLOTTED_CLASS.FIELD_30,
            SLOTTED_CLASS.FIELD_31,
            SLOTTED_CLASS.FIELD_32,
            SLOTTED_CLASS.FIELD_33,
            SLOTTED_CLASS.FIELD_34,
            SLOTTED_CLASS.FIELD_35,
            SLOTTED_CLASS.FIELD_36,
            SLOTTED_CLASS.FIELD_37,
            SLOTTED_CLASS.FIELD_38,
            SLOTTED_CLASS.FIELD_39,
            SLOTTED_CLASS.FIELD_40,
            SLOTTED_CLASS.FIELD_41,
            SLOTTED_CLASS.FIELD_42,
            SLOTTED_CLASS.FIELD_43,
            SLOTTED_CLASS.FIELD_44,
            SLOTTED_CLASS.FIELD_45,
            SLOTTED_CLASS.FIELD_46,
            SLOTTED_CLASS.FIELD_47,
            SLOTTED_CLASS.FIELD_48,
            SLOTTED_CLASS.FIELD_49,
            SLOTTED_CLASS.FIELD_50,
            SLOTTED_CLASS.FIELD_51,
            SLOTTED_CLASS.FIELD_52,
            SLOTTED_CLASS.FIELD_53,
            SLOTTED_CLASS.FIELD_54,
            SLOTTED_CLASS.FIELD_55,
            SLOTTED_CLASS.FIELD_56,
            SLOTTED_CLASS.FIELD_57,
            SLOTTED_CLASS.FIELD_58,
            SLOTTED_CLASS.FIELD_59,
            SLOTTED_CLASS.FIELD_60,
            SLOTTED_CLASS.FIELD_61,
            SLOTTED_CLASS.FIELD_62,
            SLOTTED_CLASS.FIELD_63,
            SLOTTED_CLASS.FIELD_64,
            SLOTTED_CLASS.FIELD_65,
            SLOTTED_CLASS.FIELD_66,
            SLOTTED_CLASS.FIELD_67,
            SLOTTED_CLASS.FIELD_68,
            SLOTTED_CLASS.FIELD_69,
            SLOTTED_CLASS.FIELD_70,
            SLOTTED_CLASS.FIELD_71,
            SLOTTED_CLASS.FIELD_72,
            SLOTTED_CLASS.FIELD_73,
            SLOTTED_CLASS.FIELD_74,
            SLOTTED_CLASS.FIELD_75,
            SLOTTED_CLASS.FIELD_76,
            SLOTTED_CLASS.FIELD_77,
            SLOTTED_CLASS.FIELD_78,
            SLOTTED_CLASS.FIELD_79,
            SLOTTED_CLASS.FIELD_80,
            SLOTTED_CLASS.FIELD_81,
            SLOTTED_CLASS.FIELD_82,
            SLOTTED_CLASS.FIELD_83,
            SLOTTED_CLASS.FIELD_84,
            SLOTTED_CLASS.FIELD_85,
            SLOTTED_CLASS.FIELD_86,
            SLOTTED_CLASS.FIELD_87,
            SLOTTED_CLASS.FIELD_88,
            SLOTTED_CLASS.FIELD_89,
            SLOTTED_CLASS.FIELD_90,
            SLOTTED_CLASS.FIELD_91,
            SLOTTED_CLASS.FIELD_92,
            SLOTTED_CLASS.FIELD_93,
            SLOTTED_CLASS.FIELD_94,
            SLOTTED_CLASS.FIELD_95,
            SLOTTED_CLASS.FIELD_96,
            SLOTTED_CLASS.FIELD_97,
            SLOTTED_CLASS.FIELD_98,
            SLOTTED_CLASS.FIELD_99,
            SLOTTED_CLASS.FIELD_100,
            SLOTTED_CLASS.FIELD_101,
            SLOTTED_CLASS.FIELD_102,
            SLOTTED_CLASS.FIELD_103,
            SLOTTED_CLASS.FIELD_104,
            SLOTTED_CLASS.FIELD_105,
            SLOTTED_CLASS.FIELD_106,
            SLOTTED_CLASS.FIELD_107,
            SLOTTED_CLASS.FIELD_108,
            SLOTTED_CLASS.FIELD_109,
            SLOTTED_CLASS.FIELD_110,
            SLOTTED_CLASS.FIELD_111,
            SLOTTED_CLASS.FIELD_112,
            SLOTTED_CLASS.FIELD_113,
            SLOTTED_CLASS.FIELD_114,
            SLOTTED_CLASS.FIELD_115,
            SLOTTED_CLASS.FIELD_116,
            SLOTTED_CLASS.FIELD_117,
            SLOTTED_CLASS.FIELD_118,
            SLOTTED_CLASS.FIELD_119,
            SLOTTED_CLASS.FIELD_120,
            SLOTTED_CLASS.FIELD_121,
            SLOTTED_CLASS.FIELD_122,
            SLOTTED_CLASS.FIELD_123,
            SLOTTED_CLASS.FIELD_124,
            SLOTTED_CLASS.FIELD_125,
            SLOTTED_CLASS.FIELD_126,
            SLOTTED_CLASS.FIELD_127,
        )
    )


def retrieve_plain_direct_15(_start, _n_slots):
    return (
        PlainClass.FIELD_0
        + PlainClass.FIELD_1
        + PlainClass.FIELD_2
        + PlainClass.FIELD_3
        + PlainClass.FIELD_4
        + PlainClass.FIELD_5
        + PlainClass.FIELD_6
        + PlainClass.FIELD_7
        + PlainClass.FIELD_8
        + PlainClass.FIELD_9
        + PlainClass.FIELD_10
        + PlainClass.FIELD_11
        + PlainClass.FIELD_12
        + PlainClass.FIELD_13
        + PlainClass.FIELD_14
    )


def retrieve_plain_direct_20(_start, _n_slots):
    return (
        PlainClass.FIELD_0
        + PlainClass.FIELD_1
        + PlainClass.FIELD_2
        + PlainClass.FIELD_3
        + PlainClass.FIELD_4
        + PlainClass.FIELD_5
        + PlainClass.FIELD_6
        + PlainClass.FIELD_7
        + PlainClass.FIELD_8
        + PlainClass.FIELD_9
        + PlainClass.FIELD_10
        + PlainClass.FIELD_11
        + PlainClass.FIELD_12
        + PlainClass.FIELD_13
        + PlainClass.FIELD_14
        + PlainClass.FIELD_15
        + PlainClass.FIELD_16
        + PlainClass.FIELD_17
        + PlainClass.FIELD_18
        + PlainClass.FIELD_19
    )


def retrieve_plain_direct_128(_start, _n_slots):
    return sum(
        (
            PlainClass.FIELD_0,
            PlainClass.FIELD_1,
            PlainClass.FIELD_2,
            PlainClass.FIELD_3,
            PlainClass.FIELD_4,
            PlainClass.FIELD_5,
            PlainClass.FIELD_6,
            PlainClass.FIELD_7,
            PlainClass.FIELD_8,
            PlainClass.FIELD_9,
            PlainClass.FIELD_10,
            PlainClass.FIELD_11,
            PlainClass.FIELD_12,
            PlainClass.FIELD_13,
            PlainClass.FIELD_14,
            PlainClass.FIELD_15,
            PlainClass.FIELD_16,
            PlainClass.FIELD_17,
            PlainClass.FIELD_18,
            PlainClass.FIELD_19,
            PlainClass.FIELD_20,
            PlainClass.FIELD_21,
            PlainClass.FIELD_22,
            PlainClass.FIELD_23,
            PlainClass.FIELD_24,
            PlainClass.FIELD_25,
            PlainClass.FIELD_26,
            PlainClass.FIELD_27,
            PlainClass.FIELD_28,
            PlainClass.FIELD_29,
            PlainClass.FIELD_30,
            PlainClass.FIELD_31,
            PlainClass.FIELD_32,
            PlainClass.FIELD_33,
            PlainClass.FIELD_34,
            PlainClass.FIELD_35,
            PlainClass.FIELD_36,
            PlainClass.FIELD_37,
            PlainClass.FIELD_38,
            PlainClass.FIELD_39,
            PlainClass.FIELD_40,
            PlainClass.FIELD_41,
            PlainClass.FIELD_42,
            PlainClass.FIELD_43,
            PlainClass.FIELD_44,
            PlainClass.FIELD_45,
            PlainClass.FIELD_46,
            PlainClass.FIELD_47,
            PlainClass.FIELD_48,
            PlainClass.FIELD_49,
            PlainClass.FIELD_50,
            PlainClass.FIELD_51,
            PlainClass.FIELD_52,
            PlainClass.FIELD_53,
            PlainClass.FIELD_54,
            PlainClass.FIELD_55,
            PlainClass.FIELD_56,
            PlainClass.FIELD_57,
            PlainClass.FIELD_58,
            PlainClass.FIELD_59,
            PlainClass.FIELD_60,
            PlainClass.FIELD_61,
            PlainClass.FIELD_62,
            PlainClass.FIELD_63,
            PlainClass.FIELD_64,
            PlainClass.FIELD_65,
            PlainClass.FIELD_66,
            PlainClass.FIELD_67,
            PlainClass.FIELD_68,
            PlainClass.FIELD_69,
            PlainClass.FIELD_70,
            PlainClass.FIELD_71,
            PlainClass.FIELD_72,
            PlainClass.FIELD_73,
            PlainClass.FIELD_74,
            PlainClass.FIELD_75,
            PlainClass.FIELD_76,
            PlainClass.FIELD_77,
            PlainClass.FIELD_78,
            PlainClass.FIELD_79,
            PlainClass.FIELD_80,
            PlainClass.FIELD_81,
            PlainClass.FIELD_82,
            PlainClass.FIELD_83,
            PlainClass.FIELD_84,
            PlainClass.FIELD_85,
            PlainClass.FIELD_86,
            PlainClass.FIELD_87,
            PlainClass.FIELD_88,
            PlainClass.FIELD_89,
            PlainClass.FIELD_90,
            PlainClass.FIELD_91,
            PlainClass.FIELD_92,
            PlainClass.FIELD_93,
            PlainClass.FIELD_94,
            PlainClass.FIELD_95,
            PlainClass.FIELD_96,
            PlainClass.FIELD_97,
            PlainClass.FIELD_98,
            PlainClass.FIELD_99,
            PlainClass.FIELD_100,
            PlainClass.FIELD_101,
            PlainClass.FIELD_102,
            PlainClass.FIELD_103,
            PlainClass.FIELD_104,
            PlainClass.FIELD_105,
            PlainClass.FIELD_106,
            PlainClass.FIELD_107,
            PlainClass.FIELD_108,
            PlainClass.FIELD_109,
            PlainClass.FIELD_110,
            PlainClass.FIELD_111,
            PlainClass.FIELD_112,
            PlainClass.FIELD_113,
            PlainClass.FIELD_114,
            PlainClass.FIELD_115,
            PlainClass.FIELD_116,
            PlainClass.FIELD_117,
            PlainClass.FIELD_118,
            PlainClass.FIELD_119,
            PlainClass.FIELD_120,
            PlainClass.FIELD_121,
            PlainClass.FIELD_122,
            PlainClass.FIELD_123,
            PlainClass.FIELD_124,
            PlainClass.FIELD_125,
            PlainClass.FIELD_126,
            PlainClass.FIELD_127,
        )
    )


def retrieve_classvars_direct_15(_start, _n_slots):
    return (
        CV.FIELD_0
        + CV.FIELD_1
        + CV.FIELD_2
        + CV.FIELD_3
        + CV.FIELD_4
        + CV.FIELD_5
        + CV.FIELD_6
        + CV.FIELD_7
        + CV.FIELD_8
        + CV.FIELD_9
        + CV.FIELD_10
        + CV.FIELD_11
        + CV.FIELD_12
        + CV.FIELD_13
        + CV.FIELD_14
    )


def retrieve_classvars_direct_20(_start, _n_slots):
    return (
        CV.FIELD_0
        + CV.FIELD_1
        + CV.FIELD_2
        + CV.FIELD_3
        + CV.FIELD_4
        + CV.FIELD_5
        + CV.FIELD_6
        + CV.FIELD_7
        + CV.FIELD_8
        + CV.FIELD_9
        + CV.FIELD_10
        + CV.FIELD_11
        + CV.FIELD_12
        + CV.FIELD_13
        + CV.FIELD_14
        + CV.FIELD_15
        + CV.FIELD_16
        + CV.FIELD_17
        + CV.FIELD_18
        + CV.FIELD_19
    )


def retrieve_classvars_direct_128(_start, _n_slots):
    return sum(
        (
            CV.FIELD_0,
            CV.FIELD_1,
            CV.FIELD_2,
            CV.FIELD_3,
            CV.FIELD_4,
            CV.FIELD_5,
            CV.FIELD_6,
            CV.FIELD_7,
            CV.FIELD_8,
            CV.FIELD_9,
            CV.FIELD_10,
            CV.FIELD_11,
            CV.FIELD_12,
            CV.FIELD_13,
            CV.FIELD_14,
            CV.FIELD_15,
            CV.FIELD_16,
            CV.FIELD_17,
            CV.FIELD_18,
            CV.FIELD_19,
            CV.FIELD_20,
            CV.FIELD_21,
            CV.FIELD_22,
            CV.FIELD_23,
            CV.FIELD_24,
            CV.FIELD_25,
            CV.FIELD_26,
            CV.FIELD_27,
            CV.FIELD_28,
            CV.FIELD_29,
            CV.FIELD_30,
            CV.FIELD_31,
            CV.FIELD_32,
            CV.FIELD_33,
            CV.FIELD_34,
            CV.FIELD_35,
            CV.FIELD_36,
            CV.FIELD_37,
            CV.FIELD_38,
            CV.FIELD_39,
            CV.FIELD_40,
            CV.FIELD_41,
            CV.FIELD_42,
            CV.FIELD_43,
            CV.FIELD_44,
            CV.FIELD_45,
            CV.FIELD_46,
            CV.FIELD_47,
            CV.FIELD_48,
            CV.FIELD_49,
            CV.FIELD_50,
            CV.FIELD_51,
            CV.FIELD_52,
            CV.FIELD_53,
            CV.FIELD_54,
            CV.FIELD_55,
            CV.FIELD_56,
            CV.FIELD_57,
            CV.FIELD_58,
            CV.FIELD_59,
            CV.FIELD_60,
            CV.FIELD_61,
            CV.FIELD_62,
            CV.FIELD_63,
            CV.FIELD_64,
            CV.FIELD_65,
            CV.FIELD_66,
            CV.FIELD_67,
            CV.FIELD_68,
            CV.FIELD_69,
            CV.FIELD_70,
            CV.FIELD_71,
            CV.FIELD_72,
            CV.FIELD_73,
            CV.FIELD_74,
            CV.FIELD_75,
            CV.FIELD_76,
            CV.FIELD_77,
            CV.FIELD_78,
            CV.FIELD_79,
            CV.FIELD_80,
            CV.FIELD_81,
            CV.FIELD_82,
            CV.FIELD_83,
            CV.FIELD_84,
            CV.FIELD_85,
            CV.FIELD_86,
            CV.FIELD_87,
            CV.FIELD_88,
            CV.FIELD_89,
            CV.FIELD_90,
            CV.FIELD_91,
            CV.FIELD_92,
            CV.FIELD_93,
            CV.FIELD_94,
            CV.FIELD_95,
            CV.FIELD_96,
            CV.FIELD_97,
            CV.FIELD_98,
            CV.FIELD_99,
            CV.FIELD_100,
            CV.FIELD_101,
            CV.FIELD_102,
            CV.FIELD_103,
            CV.FIELD_104,
            CV.FIELD_105,
            CV.FIELD_106,
            CV.FIELD_107,
            CV.FIELD_108,
            CV.FIELD_109,
            CV.FIELD_110,
            CV.FIELD_111,
            CV.FIELD_112,
            CV.FIELD_113,
            CV.FIELD_114,
            CV.FIELD_115,
            CV.FIELD_116,
            CV.FIELD_117,
            CV.FIELD_118,
            CV.FIELD_119,
            CV.FIELD_120,
            CV.FIELD_121,
            CV.FIELD_122,
            CV.FIELD_123,
            CV.FIELD_124,
            CV.FIELD_125,
            CV.FIELD_126,
            CV.FIELD_127,
        )
    )


def retrieve_dict_direct_15(_start, _n_slots):
    return (
        NAMED_DICT["FIELD_0"]
        + NAMED_DICT["FIELD_1"]
        + NAMED_DICT["FIELD_2"]
        + NAMED_DICT["FIELD_3"]
        + NAMED_DICT["FIELD_4"]
        + NAMED_DICT["FIELD_5"]
        + NAMED_DICT["FIELD_6"]
        + NAMED_DICT["FIELD_7"]
        + NAMED_DICT["FIELD_8"]
        + NAMED_DICT["FIELD_9"]
        + NAMED_DICT["FIELD_10"]
        + NAMED_DICT["FIELD_11"]
        + NAMED_DICT["FIELD_12"]
        + NAMED_DICT["FIELD_13"]
        + NAMED_DICT["FIELD_14"]
    )


def retrieve_dict_direct_20(_start, _n_slots):
    return (
        NAMED_DICT["FIELD_0"]
        + NAMED_DICT["FIELD_1"]
        + NAMED_DICT["FIELD_2"]
        + NAMED_DICT["FIELD_3"]
        + NAMED_DICT["FIELD_4"]
        + NAMED_DICT["FIELD_5"]
        + NAMED_DICT["FIELD_6"]
        + NAMED_DICT["FIELD_7"]
        + NAMED_DICT["FIELD_8"]
        + NAMED_DICT["FIELD_9"]
        + NAMED_DICT["FIELD_10"]
        + NAMED_DICT["FIELD_11"]
        + NAMED_DICT["FIELD_12"]
        + NAMED_DICT["FIELD_13"]
        + NAMED_DICT["FIELD_14"]
        + NAMED_DICT["FIELD_15"]
        + NAMED_DICT["FIELD_16"]
        + NAMED_DICT["FIELD_17"]
        + NAMED_DICT["FIELD_18"]
        + NAMED_DICT["FIELD_19"]
    )


def retrieve_dict_direct_128(_start, _n_slots):
    return sum(
        (
            NAMED_DICT["FIELD_0"],
            NAMED_DICT["FIELD_1"],
            NAMED_DICT["FIELD_2"],
            NAMED_DICT["FIELD_3"],
            NAMED_DICT["FIELD_4"],
            NAMED_DICT["FIELD_5"],
            NAMED_DICT["FIELD_6"],
            NAMED_DICT["FIELD_7"],
            NAMED_DICT["FIELD_8"],
            NAMED_DICT["FIELD_9"],
            NAMED_DICT["FIELD_10"],
            NAMED_DICT["FIELD_11"],
            NAMED_DICT["FIELD_12"],
            NAMED_DICT["FIELD_13"],
            NAMED_DICT["FIELD_14"],
            NAMED_DICT["FIELD_15"],
            NAMED_DICT["FIELD_16"],
            NAMED_DICT["FIELD_17"],
            NAMED_DICT["FIELD_18"],
            NAMED_DICT["FIELD_19"],
            NAMED_DICT["FIELD_20"],
            NAMED_DICT["FIELD_21"],
            NAMED_DICT["FIELD_22"],
            NAMED_DICT["FIELD_23"],
            NAMED_DICT["FIELD_24"],
            NAMED_DICT["FIELD_25"],
            NAMED_DICT["FIELD_26"],
            NAMED_DICT["FIELD_27"],
            NAMED_DICT["FIELD_28"],
            NAMED_DICT["FIELD_29"],
            NAMED_DICT["FIELD_30"],
            NAMED_DICT["FIELD_31"],
            NAMED_DICT["FIELD_32"],
            NAMED_DICT["FIELD_33"],
            NAMED_DICT["FIELD_34"],
            NAMED_DICT["FIELD_35"],
            NAMED_DICT["FIELD_36"],
            NAMED_DICT["FIELD_37"],
            NAMED_DICT["FIELD_38"],
            NAMED_DICT["FIELD_39"],
            NAMED_DICT["FIELD_40"],
            NAMED_DICT["FIELD_41"],
            NAMED_DICT["FIELD_42"],
            NAMED_DICT["FIELD_43"],
            NAMED_DICT["FIELD_44"],
            NAMED_DICT["FIELD_45"],
            NAMED_DICT["FIELD_46"],
            NAMED_DICT["FIELD_47"],
            NAMED_DICT["FIELD_48"],
            NAMED_DICT["FIELD_49"],
            NAMED_DICT["FIELD_50"],
            NAMED_DICT["FIELD_51"],
            NAMED_DICT["FIELD_52"],
            NAMED_DICT["FIELD_53"],
            NAMED_DICT["FIELD_54"],
            NAMED_DICT["FIELD_55"],
            NAMED_DICT["FIELD_56"],
            NAMED_DICT["FIELD_57"],
            NAMED_DICT["FIELD_58"],
            NAMED_DICT["FIELD_59"],
            NAMED_DICT["FIELD_60"],
            NAMED_DICT["FIELD_61"],
            NAMED_DICT["FIELD_62"],
            NAMED_DICT["FIELD_63"],
            NAMED_DICT["FIELD_64"],
            NAMED_DICT["FIELD_65"],
            NAMED_DICT["FIELD_66"],
            NAMED_DICT["FIELD_67"],
            NAMED_DICT["FIELD_68"],
            NAMED_DICT["FIELD_69"],
            NAMED_DICT["FIELD_70"],
            NAMED_DICT["FIELD_71"],
            NAMED_DICT["FIELD_72"],
            NAMED_DICT["FIELD_73"],
            NAMED_DICT["FIELD_74"],
            NAMED_DICT["FIELD_75"],
            NAMED_DICT["FIELD_76"],
            NAMED_DICT["FIELD_77"],
            NAMED_DICT["FIELD_78"],
            NAMED_DICT["FIELD_79"],
            NAMED_DICT["FIELD_80"],
            NAMED_DICT["FIELD_81"],
            NAMED_DICT["FIELD_82"],
            NAMED_DICT["FIELD_83"],
            NAMED_DICT["FIELD_84"],
            NAMED_DICT["FIELD_85"],
            NAMED_DICT["FIELD_86"],
            NAMED_DICT["FIELD_87"],
            NAMED_DICT["FIELD_88"],
            NAMED_DICT["FIELD_89"],
            NAMED_DICT["FIELD_90"],
            NAMED_DICT["FIELD_91"],
            NAMED_DICT["FIELD_92"],
            NAMED_DICT["FIELD_93"],
            NAMED_DICT["FIELD_94"],
            NAMED_DICT["FIELD_95"],
            NAMED_DICT["FIELD_96"],
            NAMED_DICT["FIELD_97"],
            NAMED_DICT["FIELD_98"],
            NAMED_DICT["FIELD_99"],
            NAMED_DICT["FIELD_100"],
            NAMED_DICT["FIELD_101"],
            NAMED_DICT["FIELD_102"],
            NAMED_DICT["FIELD_103"],
            NAMED_DICT["FIELD_104"],
            NAMED_DICT["FIELD_105"],
            NAMED_DICT["FIELD_106"],
            NAMED_DICT["FIELD_107"],
            NAMED_DICT["FIELD_108"],
            NAMED_DICT["FIELD_109"],
            NAMED_DICT["FIELD_110"],
            NAMED_DICT["FIELD_111"],
            NAMED_DICT["FIELD_112"],
            NAMED_DICT["FIELD_113"],
            NAMED_DICT["FIELD_114"],
            NAMED_DICT["FIELD_115"],
            NAMED_DICT["FIELD_116"],
            NAMED_DICT["FIELD_117"],
            NAMED_DICT["FIELD_118"],
            NAMED_DICT["FIELD_119"],
            NAMED_DICT["FIELD_120"],
            NAMED_DICT["FIELD_121"],
            NAMED_DICT["FIELD_122"],
            NAMED_DICT["FIELD_123"],
            NAMED_DICT["FIELD_124"],
            NAMED_DICT["FIELD_125"],
            NAMED_DICT["FIELD_126"],
            NAMED_DICT["FIELD_127"],
        )
    )


def retrieve_enum_direct_15(_start, _n_slots):
    return (
        EnumFields.FIELD_0.value
        + EnumFields.FIELD_1.value
        + EnumFields.FIELD_2.value
        + EnumFields.FIELD_3.value
        + EnumFields.FIELD_4.value
        + EnumFields.FIELD_5.value
        + EnumFields.FIELD_6.value
        + EnumFields.FIELD_7.value
        + EnumFields.FIELD_8.value
        + EnumFields.FIELD_9.value
        + EnumFields.FIELD_10.value
        + EnumFields.FIELD_11.value
        + EnumFields.FIELD_12.value
        + EnumFields.FIELD_13.value
        + EnumFields.FIELD_14.value
    )


def retrieve_enum_direct_20(_start, _n_slots):
    return (
        EnumFields.FIELD_0.value
        + EnumFields.FIELD_1.value
        + EnumFields.FIELD_2.value
        + EnumFields.FIELD_3.value
        + EnumFields.FIELD_4.value
        + EnumFields.FIELD_5.value
        + EnumFields.FIELD_6.value
        + EnumFields.FIELD_7.value
        + EnumFields.FIELD_8.value
        + EnumFields.FIELD_9.value
        + EnumFields.FIELD_10.value
        + EnumFields.FIELD_11.value
        + EnumFields.FIELD_12.value
        + EnumFields.FIELD_13.value
        + EnumFields.FIELD_14.value
        + EnumFields.FIELD_15.value
        + EnumFields.FIELD_16.value
        + EnumFields.FIELD_17.value
        + EnumFields.FIELD_18.value
        + EnumFields.FIELD_19.value
    )


def retrieve_enum_direct_128(_start, _n_slots):
    return sum(
        (
            EnumFields.FIELD_0.value,
            EnumFields.FIELD_1.value,
            EnumFields.FIELD_2.value,
            EnumFields.FIELD_3.value,
            EnumFields.FIELD_4.value,
            EnumFields.FIELD_5.value,
            EnumFields.FIELD_6.value,
            EnumFields.FIELD_7.value,
            EnumFields.FIELD_8.value,
            EnumFields.FIELD_9.value,
            EnumFields.FIELD_10.value,
            EnumFields.FIELD_11.value,
            EnumFields.FIELD_12.value,
            EnumFields.FIELD_13.value,
            EnumFields.FIELD_14.value,
            EnumFields.FIELD_15.value,
            EnumFields.FIELD_16.value,
            EnumFields.FIELD_17.value,
            EnumFields.FIELD_18.value,
            EnumFields.FIELD_19.value,
            EnumFields.FIELD_20.value,
            EnumFields.FIELD_21.value,
            EnumFields.FIELD_22.value,
            EnumFields.FIELD_23.value,
            EnumFields.FIELD_24.value,
            EnumFields.FIELD_25.value,
            EnumFields.FIELD_26.value,
            EnumFields.FIELD_27.value,
            EnumFields.FIELD_28.value,
            EnumFields.FIELD_29.value,
            EnumFields.FIELD_30.value,
            EnumFields.FIELD_31.value,
            EnumFields.FIELD_32.value,
            EnumFields.FIELD_33.value,
            EnumFields.FIELD_34.value,
            EnumFields.FIELD_35.value,
            EnumFields.FIELD_36.value,
            EnumFields.FIELD_37.value,
            EnumFields.FIELD_38.value,
            EnumFields.FIELD_39.value,
            EnumFields.FIELD_40.value,
            EnumFields.FIELD_41.value,
            EnumFields.FIELD_42.value,
            EnumFields.FIELD_43.value,
            EnumFields.FIELD_44.value,
            EnumFields.FIELD_45.value,
            EnumFields.FIELD_46.value,
            EnumFields.FIELD_47.value,
            EnumFields.FIELD_48.value,
            EnumFields.FIELD_49.value,
            EnumFields.FIELD_50.value,
            EnumFields.FIELD_51.value,
            EnumFields.FIELD_52.value,
            EnumFields.FIELD_53.value,
            EnumFields.FIELD_54.value,
            EnumFields.FIELD_55.value,
            EnumFields.FIELD_56.value,
            EnumFields.FIELD_57.value,
            EnumFields.FIELD_58.value,
            EnumFields.FIELD_59.value,
            EnumFields.FIELD_60.value,
            EnumFields.FIELD_61.value,
            EnumFields.FIELD_62.value,
            EnumFields.FIELD_63.value,
            EnumFields.FIELD_64.value,
            EnumFields.FIELD_65.value,
            EnumFields.FIELD_66.value,
            EnumFields.FIELD_67.value,
            EnumFields.FIELD_68.value,
            EnumFields.FIELD_69.value,
            EnumFields.FIELD_70.value,
            EnumFields.FIELD_71.value,
            EnumFields.FIELD_72.value,
            EnumFields.FIELD_73.value,
            EnumFields.FIELD_74.value,
            EnumFields.FIELD_75.value,
            EnumFields.FIELD_76.value,
            EnumFields.FIELD_77.value,
            EnumFields.FIELD_78.value,
            EnumFields.FIELD_79.value,
            EnumFields.FIELD_80.value,
            EnumFields.FIELD_81.value,
            EnumFields.FIELD_82.value,
            EnumFields.FIELD_83.value,
            EnumFields.FIELD_84.value,
            EnumFields.FIELD_85.value,
            EnumFields.FIELD_86.value,
            EnumFields.FIELD_87.value,
            EnumFields.FIELD_88.value,
            EnumFields.FIELD_89.value,
            EnumFields.FIELD_90.value,
            EnumFields.FIELD_91.value,
            EnumFields.FIELD_92.value,
            EnumFields.FIELD_93.value,
            EnumFields.FIELD_94.value,
            EnumFields.FIELD_95.value,
            EnumFields.FIELD_96.value,
            EnumFields.FIELD_97.value,
            EnumFields.FIELD_98.value,
            EnumFields.FIELD_99.value,
            EnumFields.FIELD_100.value,
            EnumFields.FIELD_101.value,
            EnumFields.FIELD_102.value,
            EnumFields.FIELD_103.value,
            EnumFields.FIELD_104.value,
            EnumFields.FIELD_105.value,
            EnumFields.FIELD_106.value,
            EnumFields.FIELD_107.value,
            EnumFields.FIELD_108.value,
            EnumFields.FIELD_109.value,
            EnumFields.FIELD_110.value,
            EnumFields.FIELD_111.value,
            EnumFields.FIELD_112.value,
            EnumFields.FIELD_113.value,
            EnumFields.FIELD_114.value,
            EnumFields.FIELD_115.value,
            EnumFields.FIELD_116.value,
            EnumFields.FIELD_117.value,
            EnumFields.FIELD_118.value,
            EnumFields.FIELD_119.value,
            EnumFields.FIELD_120.value,
            EnumFields.FIELD_121.value,
            EnumFields.FIELD_122.value,
            EnumFields.FIELD_123.value,
            EnumFields.FIELD_124.value,
            EnumFields.FIELD_125.value,
            EnumFields.FIELD_126.value,
            EnumFields.FIELD_127.value,
        )
    )


def retrieve_intenum_direct_15(_start, _n_slots):
    return (
        IntFields.FIELD_0
        + IntFields.FIELD_1
        + IntFields.FIELD_2
        + IntFields.FIELD_3
        + IntFields.FIELD_4
        + IntFields.FIELD_5
        + IntFields.FIELD_6
        + IntFields.FIELD_7
        + IntFields.FIELD_8
        + IntFields.FIELD_9
        + IntFields.FIELD_10
        + IntFields.FIELD_11
        + IntFields.FIELD_12
        + IntFields.FIELD_13
        + IntFields.FIELD_14
    )


def retrieve_intenum_direct_20(_start, _n_slots):
    return (
        IntFields.FIELD_0
        + IntFields.FIELD_1
        + IntFields.FIELD_2
        + IntFields.FIELD_3
        + IntFields.FIELD_4
        + IntFields.FIELD_5
        + IntFields.FIELD_6
        + IntFields.FIELD_7
        + IntFields.FIELD_8
        + IntFields.FIELD_9
        + IntFields.FIELD_10
        + IntFields.FIELD_11
        + IntFields.FIELD_12
        + IntFields.FIELD_13
        + IntFields.FIELD_14
        + IntFields.FIELD_15
        + IntFields.FIELD_16
        + IntFields.FIELD_17
        + IntFields.FIELD_18
        + IntFields.FIELD_19
    )


def retrieve_intenum_direct_128(_start, _n_slots):
    return sum(
        (
            IntFields.FIELD_0,
            IntFields.FIELD_1,
            IntFields.FIELD_2,
            IntFields.FIELD_3,
            IntFields.FIELD_4,
            IntFields.FIELD_5,
            IntFields.FIELD_6,
            IntFields.FIELD_7,
            IntFields.FIELD_8,
            IntFields.FIELD_9,
            IntFields.FIELD_10,
            IntFields.FIELD_11,
            IntFields.FIELD_12,
            IntFields.FIELD_13,
            IntFields.FIELD_14,
            IntFields.FIELD_15,
            IntFields.FIELD_16,
            IntFields.FIELD_17,
            IntFields.FIELD_18,
            IntFields.FIELD_19,
            IntFields.FIELD_20,
            IntFields.FIELD_21,
            IntFields.FIELD_22,
            IntFields.FIELD_23,
            IntFields.FIELD_24,
            IntFields.FIELD_25,
            IntFields.FIELD_26,
            IntFields.FIELD_27,
            IntFields.FIELD_28,
            IntFields.FIELD_29,
            IntFields.FIELD_30,
            IntFields.FIELD_31,
            IntFields.FIELD_32,
            IntFields.FIELD_33,
            IntFields.FIELD_34,
            IntFields.FIELD_35,
            IntFields.FIELD_36,
            IntFields.FIELD_37,
            IntFields.FIELD_38,
            IntFields.FIELD_39,
            IntFields.FIELD_40,
            IntFields.FIELD_41,
            IntFields.FIELD_42,
            IntFields.FIELD_43,
            IntFields.FIELD_44,
            IntFields.FIELD_45,
            IntFields.FIELD_46,
            IntFields.FIELD_47,
            IntFields.FIELD_48,
            IntFields.FIELD_49,
            IntFields.FIELD_50,
            IntFields.FIELD_51,
            IntFields.FIELD_52,
            IntFields.FIELD_53,
            IntFields.FIELD_54,
            IntFields.FIELD_55,
            IntFields.FIELD_56,
            IntFields.FIELD_57,
            IntFields.FIELD_58,
            IntFields.FIELD_59,
            IntFields.FIELD_60,
            IntFields.FIELD_61,
            IntFields.FIELD_62,
            IntFields.FIELD_63,
            IntFields.FIELD_64,
            IntFields.FIELD_65,
            IntFields.FIELD_66,
            IntFields.FIELD_67,
            IntFields.FIELD_68,
            IntFields.FIELD_69,
            IntFields.FIELD_70,
            IntFields.FIELD_71,
            IntFields.FIELD_72,
            IntFields.FIELD_73,
            IntFields.FIELD_74,
            IntFields.FIELD_75,
            IntFields.FIELD_76,
            IntFields.FIELD_77,
            IntFields.FIELD_78,
            IntFields.FIELD_79,
            IntFields.FIELD_80,
            IntFields.FIELD_81,
            IntFields.FIELD_82,
            IntFields.FIELD_83,
            IntFields.FIELD_84,
            IntFields.FIELD_85,
            IntFields.FIELD_86,
            IntFields.FIELD_87,
            IntFields.FIELD_88,
            IntFields.FIELD_89,
            IntFields.FIELD_90,
            IntFields.FIELD_91,
            IntFields.FIELD_92,
            IntFields.FIELD_93,
            IntFields.FIELD_94,
            IntFields.FIELD_95,
            IntFields.FIELD_96,
            IntFields.FIELD_97,
            IntFields.FIELD_98,
            IntFields.FIELD_99,
            IntFields.FIELD_100,
            IntFields.FIELD_101,
            IntFields.FIELD_102,
            IntFields.FIELD_103,
            IntFields.FIELD_104,
            IntFields.FIELD_105,
            IntFields.FIELD_106,
            IntFields.FIELD_107,
            IntFields.FIELD_108,
            IntFields.FIELD_109,
            IntFields.FIELD_110,
            IntFields.FIELD_111,
            IntFields.FIELD_112,
            IntFields.FIELD_113,
            IntFields.FIELD_114,
            IntFields.FIELD_115,
            IntFields.FIELD_116,
            IntFields.FIELD_117,
            IntFields.FIELD_118,
            IntFields.FIELD_119,
            IntFields.FIELD_120,
            IntFields.FIELD_121,
            IntFields.FIELD_122,
            IntFields.FIELD_123,
            IntFields.FIELD_124,
            IntFields.FIELD_125,
            IntFields.FIELD_126,
            IntFields.FIELD_127,
        )
    )


def retrieve_namedtuple_direct_15(_start, _n_slots):
    return (
        NAMED_TUPLE.FIELD_0
        + NAMED_TUPLE.FIELD_1
        + NAMED_TUPLE.FIELD_2
        + NAMED_TUPLE.FIELD_3
        + NAMED_TUPLE.FIELD_4
        + NAMED_TUPLE.FIELD_5
        + NAMED_TUPLE.FIELD_6
        + NAMED_TUPLE.FIELD_7
        + NAMED_TUPLE.FIELD_8
        + NAMED_TUPLE.FIELD_9
        + NAMED_TUPLE.FIELD_10
        + NAMED_TUPLE.FIELD_11
        + NAMED_TUPLE.FIELD_12
        + NAMED_TUPLE.FIELD_13
        + NAMED_TUPLE.FIELD_14
    )


def retrieve_namedtuple_direct_20(_start, _n_slots):
    return (
        NAMED_TUPLE.FIELD_0
        + NAMED_TUPLE.FIELD_1
        + NAMED_TUPLE.FIELD_2
        + NAMED_TUPLE.FIELD_3
        + NAMED_TUPLE.FIELD_4
        + NAMED_TUPLE.FIELD_5
        + NAMED_TUPLE.FIELD_6
        + NAMED_TUPLE.FIELD_7
        + NAMED_TUPLE.FIELD_8
        + NAMED_TUPLE.FIELD_9
        + NAMED_TUPLE.FIELD_10
        + NAMED_TUPLE.FIELD_11
        + NAMED_TUPLE.FIELD_12
        + NAMED_TUPLE.FIELD_13
        + NAMED_TUPLE.FIELD_14
        + NAMED_TUPLE.FIELD_15
        + NAMED_TUPLE.FIELD_16
        + NAMED_TUPLE.FIELD_17
        + NAMED_TUPLE.FIELD_18
        + NAMED_TUPLE.FIELD_19
    )


def retrieve_namedtuple_direct_128(_start, _n_slots):
    return sum(
        (
            NAMED_TUPLE.FIELD_0,
            NAMED_TUPLE.FIELD_1,
            NAMED_TUPLE.FIELD_2,
            NAMED_TUPLE.FIELD_3,
            NAMED_TUPLE.FIELD_4,
            NAMED_TUPLE.FIELD_5,
            NAMED_TUPLE.FIELD_6,
            NAMED_TUPLE.FIELD_7,
            NAMED_TUPLE.FIELD_8,
            NAMED_TUPLE.FIELD_9,
            NAMED_TUPLE.FIELD_10,
            NAMED_TUPLE.FIELD_11,
            NAMED_TUPLE.FIELD_12,
            NAMED_TUPLE.FIELD_13,
            NAMED_TUPLE.FIELD_14,
            NAMED_TUPLE.FIELD_15,
            NAMED_TUPLE.FIELD_16,
            NAMED_TUPLE.FIELD_17,
            NAMED_TUPLE.FIELD_18,
            NAMED_TUPLE.FIELD_19,
            NAMED_TUPLE.FIELD_20,
            NAMED_TUPLE.FIELD_21,
            NAMED_TUPLE.FIELD_22,
            NAMED_TUPLE.FIELD_23,
            NAMED_TUPLE.FIELD_24,
            NAMED_TUPLE.FIELD_25,
            NAMED_TUPLE.FIELD_26,
            NAMED_TUPLE.FIELD_27,
            NAMED_TUPLE.FIELD_28,
            NAMED_TUPLE.FIELD_29,
            NAMED_TUPLE.FIELD_30,
            NAMED_TUPLE.FIELD_31,
            NAMED_TUPLE.FIELD_32,
            NAMED_TUPLE.FIELD_33,
            NAMED_TUPLE.FIELD_34,
            NAMED_TUPLE.FIELD_35,
            NAMED_TUPLE.FIELD_36,
            NAMED_TUPLE.FIELD_37,
            NAMED_TUPLE.FIELD_38,
            NAMED_TUPLE.FIELD_39,
            NAMED_TUPLE.FIELD_40,
            NAMED_TUPLE.FIELD_41,
            NAMED_TUPLE.FIELD_42,
            NAMED_TUPLE.FIELD_43,
            NAMED_TUPLE.FIELD_44,
            NAMED_TUPLE.FIELD_45,
            NAMED_TUPLE.FIELD_46,
            NAMED_TUPLE.FIELD_47,
            NAMED_TUPLE.FIELD_48,
            NAMED_TUPLE.FIELD_49,
            NAMED_TUPLE.FIELD_50,
            NAMED_TUPLE.FIELD_51,
            NAMED_TUPLE.FIELD_52,
            NAMED_TUPLE.FIELD_53,
            NAMED_TUPLE.FIELD_54,
            NAMED_TUPLE.FIELD_55,
            NAMED_TUPLE.FIELD_56,
            NAMED_TUPLE.FIELD_57,
            NAMED_TUPLE.FIELD_58,
            NAMED_TUPLE.FIELD_59,
            NAMED_TUPLE.FIELD_60,
            NAMED_TUPLE.FIELD_61,
            NAMED_TUPLE.FIELD_62,
            NAMED_TUPLE.FIELD_63,
            NAMED_TUPLE.FIELD_64,
            NAMED_TUPLE.FIELD_65,
            NAMED_TUPLE.FIELD_66,
            NAMED_TUPLE.FIELD_67,
            NAMED_TUPLE.FIELD_68,
            NAMED_TUPLE.FIELD_69,
            NAMED_TUPLE.FIELD_70,
            NAMED_TUPLE.FIELD_71,
            NAMED_TUPLE.FIELD_72,
            NAMED_TUPLE.FIELD_73,
            NAMED_TUPLE.FIELD_74,
            NAMED_TUPLE.FIELD_75,
            NAMED_TUPLE.FIELD_76,
            NAMED_TUPLE.FIELD_77,
            NAMED_TUPLE.FIELD_78,
            NAMED_TUPLE.FIELD_79,
            NAMED_TUPLE.FIELD_80,
            NAMED_TUPLE.FIELD_81,
            NAMED_TUPLE.FIELD_82,
            NAMED_TUPLE.FIELD_83,
            NAMED_TUPLE.FIELD_84,
            NAMED_TUPLE.FIELD_85,
            NAMED_TUPLE.FIELD_86,
            NAMED_TUPLE.FIELD_87,
            NAMED_TUPLE.FIELD_88,
            NAMED_TUPLE.FIELD_89,
            NAMED_TUPLE.FIELD_90,
            NAMED_TUPLE.FIELD_91,
            NAMED_TUPLE.FIELD_92,
            NAMED_TUPLE.FIELD_93,
            NAMED_TUPLE.FIELD_94,
            NAMED_TUPLE.FIELD_95,
            NAMED_TUPLE.FIELD_96,
            NAMED_TUPLE.FIELD_97,
            NAMED_TUPLE.FIELD_98,
            NAMED_TUPLE.FIELD_99,
            NAMED_TUPLE.FIELD_100,
            NAMED_TUPLE.FIELD_101,
            NAMED_TUPLE.FIELD_102,
            NAMED_TUPLE.FIELD_103,
            NAMED_TUPLE.FIELD_104,
            NAMED_TUPLE.FIELD_105,
            NAMED_TUPLE.FIELD_106,
            NAMED_TUPLE.FIELD_107,
            NAMED_TUPLE.FIELD_108,
            NAMED_TUPLE.FIELD_109,
            NAMED_TUPLE.FIELD_110,
            NAMED_TUPLE.FIELD_111,
            NAMED_TUPLE.FIELD_112,
            NAMED_TUPLE.FIELD_113,
            NAMED_TUPLE.FIELD_114,
            NAMED_TUPLE.FIELD_115,
            NAMED_TUPLE.FIELD_116,
            NAMED_TUPLE.FIELD_117,
            NAMED_TUPLE.FIELD_118,
            NAMED_TUPLE.FIELD_119,
            NAMED_TUPLE.FIELD_120,
            NAMED_TUPLE.FIELD_121,
            NAMED_TUPLE.FIELD_122,
            NAMED_TUPLE.FIELD_123,
            NAMED_TUPLE.FIELD_124,
            NAMED_TUPLE.FIELD_125,
            NAMED_TUPLE.FIELD_126,
            NAMED_TUPLE.FIELD_127,
        )
    )


DIRECT_METHODS = {
    n_slots: (
        (
            "plain class + direct names",
            globals()[f"retrieve_plain_direct_{n_slots}"],
        ),
        (
            "classvars + direct names",
            globals()[f"retrieve_classvars_direct_{n_slots}"],
        ),
        (
            "slotted instance + direct names",
            globals()[f"retrieve_slots_direct_{n_slots}"],
        ),
        (
            "Dict + direct names",
            globals()[f"retrieve_dict_direct_{n_slots}"],
        ),
        (
            "namedtuple + direct names",
            globals()[f"retrieve_namedtuple_direct_{n_slots}"],
        ),
        (
            "Enum + direct names",
            globals()[f"retrieve_enum_direct_{n_slots}"],
        ),
        (
            "IntEnum + direct names",
            globals()[f"retrieve_intenum_direct_{n_slots}"],
        ),
    )
    for n_slots in DIRECT_SIZES
}


METHODS = (
    ("IntEnum[name].value", retrieve_int_enum),
    ("Enum[name].value", retrieve_enum),
    ("plain class + getattr", retrieve_plain_class),
    ("slotted instance + getattr", retrieve_slots),
    ("named dict[name]", retrieve_named_dict),
    ("named tuple + getattr", retrieve_named_tuple),
)


def run_check(cycles, n_slots):
    expected = n_slots * (n_slots - 1) // 2
    methods = METHODS + DIRECT_METHODS.get(n_slots, ())
    for _, method in methods:
        assert sum(method(0, n_slots) for _ in range(cycles % 17)) >= 0
        assert method(0, n_slots) == expected


def benchmark(method, cycles, n_slots):
    began = time.perf_counter()
    total = 0
    for _ in range(cycles):
        total += method(0, n_slots)
    elapsed = time.perf_counter() - began
    return elapsed, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    sizes = DIRECT_SIZES

    print(f"cycles={args.cycles:,}, repeats={args.repeats}, field_count={FIELD_COUNT}")
    for n_slots in sizes:
        run_check(args.cycles, n_slots)
        results = []
        methods = METHODS + DIRECT_METHODS.get(n_slots, ())
        for name, method in methods:
            samples = [
                benchmark(method, args.cycles, n_slots)[0] for _ in range(args.repeats)
            ]
            results.append((min(samples), name))
        fastest = min(value for value, _ in results) + 1e-20
        print(f"\n{n_slots} sequential retrievals")
        for elapsed, name in sorted(results):
            print(
                f"  {name:28} {elapsed * 1e3:9.2f} ms  {elapsed / args.cycles / n_slots * 1e9:8.2f} ns/retrieval  {elapsed / fastest:5.2f}x"
            )


if __name__ == "__main__":
    main()
