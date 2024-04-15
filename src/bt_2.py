import torch
from torch import tensor

zero_t = (0, 1, 0)
start_t = (1, 16, -1)
move_dest = (0, 16, 1)
hit_dest = (-1, 0, 2)
bar_t = (-14, 1, -1)
empty_t = (-15, 1, 0)
any_t = (-15, 16, 0)


def r():
    return range(26)


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


def noop():
    return [([], [-15 for _ in r()], [16 for _ in r()], [0 for _ in r()])]


def tensorize(x):
    moves_tensor = []
    lower_bounds = []
    upper_bounds = []
    vectors_tensor = []
    moves_dict = set()
    for moves, low, high, vector in x:
        hit_count = 0
        bearoff_count = 0
        for _, end, is_hit in zip(moves[0::3], moves[1::3], moves[2::3]):
            if is_hit:
                hit_count += 1
            if end <= 0:
                bearoff_count += 1
        assert vector[0] == -1 * hit_count
        assert sum(vector) == -1 * bearoff_count
        moves_key = tuple(moves)
        assert moves_key not in moves_dict
        moves_dict.add(moves_key)
        moves_tensor.append(moves)

        lower_bounds.append(low)
        upper_bounds.append(high)
        vectors_tensor.append(vector)
    return (
        tensor(moves_tensor, dtype=torch.float),
        tensor(lower_bounds, dtype=torch.float),
        tensor(upper_bounds, dtype=torch.float),
        tensor(vectors_tensor, dtype=torch.float),
    )
