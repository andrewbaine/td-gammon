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
    ms.append(start)
    ms.append(end)
    ms.append(1 if is_hit else 0)
    return ms


def split_out(m):
    ps = []
    qs = []
    rs = []
    for a, b, c in m:
        ps.append(a)
        qs.append(b)
        rs.append(c)
    assert len(ps) == 26
    assert len(qs) == 26
    assert len(rs) == 26
    return ps, qs, rs


def all_moves_die_start(die, start):
    assert 0 < die < 7
    assert 0 < start < 26
    end = start - die
    move = (start, end)
    if end < 0:
        # over-bearoff
        low, high, vector = split_out(
            [empty_t if i > start else start_t if i == start else any_t for i in r()]
        )
        return [([start, end, 0], low, high, vector)]
    elif end == 0:
        # exact bearoff
        low, high, vector = split_out(
            [empty_t if i > 6 else start_t if i == start else any_t for i in r()]
        )
        return [([start, end, 0], low, high, vector)]
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
            a, b, c = split_out(move)
            d, e, f = split_out(hit)
            return [
                ([start, end, 0], a, b, c),
                ([start, end, 1], d, e, f),
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
        a, b, c = split_out(move)
        s, t, u = split_out(hit)
        return [([start, end, 0], a, b, c), ([start, end, 1], s, t, u)]


def all_moves_die(die):
    result = []
    for start in range(1, 26):
        for m in all_moves_die_start(die, start):
            result.append(m)
    return result


def remove_impossible(f):
    def pred(x):
        (_, low, high, vector) = x
        assert len(vector) == 26
        for l, h in zip(low, high):
            if l >= h:
                return False
        return True

    def g(move, die, start):
        return filter(pred, f(move, die, start))

    return g


@remove_impossible
def combine_move_with_die_and_start(move, die, start):
    (moves, lows, highs, vs) = move
    end = start - die
    # over-bearoff
    move_vs = []
    hit_vs = []

    for i, t in enumerate(zip(lows, highs, vs)):
        low, high, v = t
        assert low < high
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
                t = None
                if v > 0:
                    t = (low, high, v - 1)
                else:
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

    a, b, c = split_out(move_vs)
    d, e, f = split_out(hit_vs)
    return [
        (with_move(moves, start, end, False), a, b, c),
        (with_move(moves, start, end, True), d, e, f),
    ]


def all_moves_a_b(a, b):
    result = []
    moves = []
    for s1 in range(25, 0, -1):
        for x in all_moves_die_start(a, s1):
            moves.append(x)
        start = s1
        for move in moves:
            for x in combine_move_with_die_and_start(move, b, start):
                result.append(x)
    return result


def all_moves_dice(d1, d2):
    filtered = []
    for d1, d2 in [(d1, d2), (d2, d1)]:
        for x in all_moves_a_b(d1, d2):
            (moves, high, low, vector) = x
            [s1, e1, h1, s2, e2, h2] = moves
            if s1 != s2 or e1 > e2:
                filtered.append(x)
    return filtered


def find_index(a, b):
    if a == b:
        assert False
    if a < b:
        (a, b) = (b, a)
    assert a > b
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
            return 9 + b


def tensorize(moves):
    moves_tensor = []
    lower_bounds = []
    upper_bounds = []
    vectors_tensor = []
    moves_dict = set()
    for moves, low, high, vector in moves:

        moves_key = tuple(moves)
        assert moves_key not in moves_dict
        moves_dict.add(moves_key)
        moves_tensor.append(moves)

        lower_bounds.append(low)
        upper_bounds.append(high)
        vectors_tensor.append(vector)
    return (
        tensor(moves_tensor),
        tensor(lower_bounds),
        tensor(upper_bounds),
        tensor(vectors_tensor),
    )


xs = []

i = 0
for d1 in range(1, 7):
    for d2 in range(1, d1):
        i += 1
        moves = all_moves_dice(d1, d2)
        xs.append(tensorize(moves))

assert i == 15
assert len(xs) == 15

ys = []
for die in range(1, 7):
    moves = all_moves_die(die)
    ys.append(tensorize(moves))
assert len(ys) == 6


def all_doubles(d):
    all_ss = set()
    moves_caches = [set() for i in range(4)]

    moves = []
    moves_2 = []
    moves_3 = []
    moves_4 = []
    for s1 in range(25, 0, -1):
        moves_cache = moves_caches[0]
        mm1 = []
        for x in all_moves_die_start(d, s1):
            (ms, _, __, ___) = x
            assert len(ms) == 3
            key = tuple(ms)
            assert key not in moves_cache
            moves_cache.add(key)
            moves.append(x)
            mm1.append(x)
        for s2 in range(s1, 0, -1):
            moves_cache = moves_caches[1]
            mm2 = []
            for m1 in mm1:
                for x in combine_move_with_die_and_start(m1, d, s2):
                    (ms, _, __, ___) = x
                    assert len(ms) == 6
                    key = tuple(ms)
                    assert key not in moves_cache
                    moves_cache.add(key)
                    moves_2.append(x)
                    mm2.append(x)
            for s3 in range(s2, 0, -1):
                moves_cache = moves_caches[2]
                mm3 = []
                for m2 in mm2:
                    for x in combine_move_with_die_and_start(m2, d, s3):
                        (ms, _, __, ___) = x
                        assert len(ms) == 9
                        key = tuple(ms)
                        assert key not in moves_cache
                        moves_cache.add(key)
                        moves_3.append(x)
                        mm3.append(x)
                for s4 in range(s3, 0, -1):
                    moves_cache = moves_caches[3]
                    assert (s1, s2, s3, s4) not in all_ss
                    all_ss.add((s1, s2, s3, s4))
                    for m3 in mm3:
                        for x in combine_move_with_die_and_start(m3, d, s4):
                            (ms, _, __, ___) = x
                            assert len(ms) == 12
                            key = tuple(ms)
                            assert key not in moves_cache
                            moves_cache.add(key)
                            moves_4.append(x)
    return (moves, moves_2, moves_3, moves_4)


dubs_4 = []
dubs_3 = []
dubs_2 = []
dubs_1 = []
for die in range(1, 7):
    (a, b, c, d) = all_doubles(die)
    dubs_1.append(tensorize(a))
    dubs_2.append(tensorize(b))
    dubs_3.append(tensorize(c))
    dubs_4.append(tensorize(d))


def dubs(board, die):
    i = die - 1
    (moves, lower, upper, vector) = dubs_4[i]
    indices = torch.all(lower <= board, dim=1) & torch.all(upper > board, dim=1)
    ms = moves[indices]
    if ms.size()[0] == 0:
        (moves, lower, upper, vector) = dubs_3[i]
        indices = torch.all(lower <= board, dim=1) & torch.all(upper > board, dim=1)
        ms = moves[indices]
        if ms.size()[0] == 0:
            (moves, lower, upper, vector) = dubs_2[i]
            indices = torch.all(lower <= board, dim=1) & torch.all(upper > board, dim=1)
            ms = moves[indices]
            if ms.size()[0] == 0:
                (moves, lower, upper, vector) = dubs_1[i]
                indices = torch.all(lower <= board, dim=1) & torch.all(
                    upper > board, dim=1
                )
                ms = moves[indices]
                return ms
            else:
                return ms

        else:
            return ms
    else:
        return ms


def compute_moves(board, dice):
    d1, d2 = dice
    if d1 == d2:
        return dubs(board, d1)
    assert d1 != d2
    i = find_index(*dice)
    (moves, lower, upper, vector) = xs[i]
    indices = torch.all(lower <= board, dim=1) & torch.all(upper > board, dim=1)
    ms = moves[indices]
    if ms.size()[0] == 0:
        big, small = (d1, d2) if d1 > d2 else (d2, d1)
        (moves, lower, upper, vector) = ys[big - 1]
        indices = torch.all(lower <= board, dim=1) & torch.all(upper > board, dim=1)
        ms = moves[indices]
        if ms.size()[0] == 0:
            (moves, lower, upper, vector) = ys[small - 1]
            indices = torch.all(lower <= board, dim=1) & torch.all(upper > board, dim=1)
            ms = moves[indices]
        else:
            return ms

    else:
        return ms
