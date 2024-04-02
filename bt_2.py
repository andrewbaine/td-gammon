import torch
from torch import tensor

zero_t = (0, 1, 0)
start_t = (1, 16, -1)
move_dest = (0, 16, 1)
hit_dest = (-1, 0, 2)
bar_t = (-14, 1, -1)
empty_t = (-15, 1, 0)
any_t = (-15, 16, 0)
impossible_t = (0, 0, 0)


def r():
    return range(26)


def with_move(moves, start, end, is_hit):
    ms = [m for m in moves]
    ms.append((start, end, is_hit))
    return ms


def all_moves_die_start(die, start):
    assert 0 < die < 7
    assert 0 < start < 26
    end = start - die
    move = (start, end)
    if end < 0:
        # over-bearoff
        move = [empty_t if i > start else start_t if i == start else any_t for i in r()]
        return [(die, [(start, end, False)], move)]
    elif end == 0:
        # exact bearoff
        move = [empty_t if i > 6 else start_t if i == start else any_t for i in r()]
        return [(die, [(start, end, False)], move)]
    else:
        assert end > 0
        if start == 25:
            move = [
                start_t if i == start else move_dest if i == end else any_t for i in r()
            ]
            hit = [
                (
                    start_t
                    if i == start
                    else hit_dest if i == end else bar_t if i == 0 else any_t
                )
                for i in r()
            ]
            return [
                (die, [(start, end, False)], move),
                (die, [(start, end, True)], hit),
            ]
        assert start != 25
        move = [
            (
                zero_t
                if i == 25
                else (start_t if i == start else move_dest if i == end else any_t)
            )
            for i in r()
        ]
        hit = [
            (
                zero_t
                if i == 25
                else (
                    start_t
                    if i == start
                    else hit_dest if i == end else bar_t if i == 0 else any_t
                )
            )
            for i in r()
        ]
        return [(die, [(start, end, False)], move), (die, [(start, end, True)], hit)]


def all_moves_die(die):
    result = []
    for start in range(1, 26):
        for m in all_moves_die_start(die, start):
            result.append(m)
    return result


def memoize(f):
    cache = {}

    def g(limit, dice):
        key = (limit, tuple(dice))
        if key not in cache:
            cache[key] = f(limit, dice)
        return cache[key]

    return g


def all_moves_dubs(die):
    assert 0 < die < 7
    assert False


def remove_impossible(f):
    def pred(x):
        (sum, moves, vector) = x
        assert len(vector) == 26
        return all(low < high for (low, high, _) in vector)

    def g(move, die, start):
        return filter(pred, f(move, die, start))

    return g


@remove_impossible
def combine_move_with_die_and_start(move, die, start):
    (sum, moves, vector) = move
    end = start - die
    # over-bearoff
    move_vs = []
    hit_vs = []
    for i, t in enumerate(vector):
        (low, high, v) = t
        if (end < 0 and i > start) or (end == 0 and i > 6):
            hit_vs.append(impossible_t)
            if v > 1:
                move_vs.append(impossible_t)
            elif v == 1:
                if low == -1:
                    assert high == 0
                    move_vs.append(t)
                else:
                    move_vs.append(impossible_t)
            elif v == 0:
                move_vs.append(empty_t)
            elif v < 0:
                move_vs.append((-v, 1 - v, v))
            else:
                assert False
        elif i > start:
            if i == 25:
                if low == -15:
                    assert high == 1
                    move_vs.append(t)
                    hit_vs.append(t)
                else:
                    assert low >= 0
                    t = (low, low + 1, v)
                    move_vs.append(t)
                    hit_vs.append(t)
            else:
                move_vs.append(t)
                hit_vs.append(t)
        elif i == start:
            if low == -1:
                assert high == 0
                if v < 1:
                    assert False
                elif v < 2:
                    move_vs.append(impossible_t)
                    hit_vs.append(impossible_t)
                else:
                    assert v >= 2
                    t = (-1, 0, v - 1)
                    move_vs.append(t)
                    hit_vs.append(t)
            elif low >= 0:
                assert high == 16
                t = (low + 1, high, v - 1)
                move_vs.append(t)
                hit_vs.append(t)
            else:
                assert low == -15
                assert high == 16
                move_vs.append(start_t)
                hit_vs.append(start_t)
        elif i > end:
            hit_vs.append(t)
            move_vs.append(t)
        elif i == end:
            is_hit_destination = low == -1 and high == 0
            is_move_destination = low == 0 and high == 16
            if is_hit_destination:
                move_vs.append((low, high, v + 1))
                hit_vs.append(impossible_t)
            elif is_move_destination:
                move_vs.append((low, high, v + 1))
                hit_vs.append(impossible_t)
            else:
                move_vs.append(move_dest)
                hit_vs.append(hit_dest)
        elif i > 0:
            assert i < end
            move_vs.append((low, high, v))
            hit_vs.append((low, high, v))
        else:
            assert i == 0
            move_vs.append((low, high, v))
            hit_vs.append((low, high, v + 1))  # put a piece on their bar
    return [
        (sum + die, with_move(moves, start, end, False), move_vs),
        (sum + die, with_move(moves, start, end, True), hit_vs),
    ]


def all_moves_a_b(a, b):
    result = []
    moves = []
    for s1 in range(25, 0, -1):
        for x in all_moves_die_start(a, s1):
            moves.append(x)
        for start in range(s1, s1 - 1, -1):
            for move in moves:
                for x in combine_move_with_die_and_start(move, b, start):
                    result.append(x)

    return result


# for d in range(1, 7):
#     for x in all_moves_die(d):
#         (die, move, vector) = x
#         assert die == d
#         print(move)
#         print("\t", [a for (a, b, c) in vector])
#         print("\t", [b for (a, b, c) in vector])
#         print("\t", [c for (a, b, c) in vector])


xs = []


def find_index(a, b):
    if a == b:
        assert False
    if a < b:
        (a, b) = (b, a)
    match a:
        case 2:  # 0
            return b - 1
        case 3:  # 1, 2
            return b
        case 4:
            return 2 + b
        case 5:
            return 5 + b
        case 6:
            return 10 + b


for d1 in range(1, 7):
    for d2 in range(1, d1):
        lengths_tensor = []
        moves_tensor = []
        lower_bounds = []
        upper_bounds = []
        vectors_tensor = []
        xs.append(
            (lengths_tensor, moves_tensor, lower_bounds, upper_bounds, vectors_tensor)
        )
        moves_dict = {}
        vectors_dict = {}
        for sum, moves, vector in all_moves_a_b(d1, d2):
            lengths_tensor.append(sum)
            assert len(moves) == 2
            [(a, b, x), (c, d, y)] = moves
            moves_tensor.append([a, b, 1 if x else 0, c, d, 1 if y else 0])

            moves_key = tuple(moves)
            print("move", moves_key)
            assert moves_key not in moves_dict
            moves_dict[moves_key] = (sum, moves, vector)

            vectors_key = tuple(vector)
            print("vector", vectors_key)
            assert vectors_key not in vectors_dict
            vectors_dict[vectors_key] = (sum, moves, vector)
            l = []
            h = []
            v = []
            for a, b, c in vector:
                l.append(a)
                h.append(b)
                v.append(c)
            lower_bounds.append(l)
            upper_bounds.append(h)
            vectors_tensor.append(v)

xs = [
    (
        tensor(lengths_tensor),
        tensor(moves_tensor),
        tensor(lower_bounds),
        tensor(upper_bounds),
        tensor(vector_tensor),
    )
    for (lengths_tensor, moves_tensor, lower_bounds, upper_bounds, vector_tensor) in xs
]

roll = (4, 2)
i = find_index(*roll)
(lengths, moves, lower, upper, vector) = xs[i]

import backgammon

board = tensor(backgammon.make_board())

indices = torch.all(lower <= board, dim=1) & torch.all(upper > board, dim=1)
ms = moves[indices]
print(ms)

# print(len(all_moves_a_b(2, 1)))
print("we rocked it")
