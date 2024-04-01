zero_t = (0, 1, 0)
start_t = (1, 16, -1)
move_t = (0, 16, 1)
hit_t = (-1, 0, 2)
bar_t = (-14, 1, -1)
empty_t = (-15, 1, 0)
any_t = (-15, 16, 0)


def r():
    return range(26)


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
                start_t if i == start else move_t if i == end else any_t for i in r()
            ]
            hit = [
                (
                    (1, 16, -1)
                    if i == start
                    else hit_t if i == end else bar_t if i == 0 else any_t
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
                else (start_t if i == start else move_t if i == end else any_t)
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
                    else hit_t if i == end else bar_t if i == 0 else any_t
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


def all_moves_a_b(a, b):
    assert 0 < a < 7
    assert 0 < b < 7
    assert a != b
    assert False


def all_moves_top(d1, d2):
    if d1 == d2:
        return all_moves_dubs(d1)
    else:
        xs = all_moves_a_b(d1, d2)
        ys = all_moves_a_b(d2, d1)


for d in range(1, 7):
    for x in all_moves_die(d):
        (die, move, vector) = x
        assert die == d
        print(move)
        print("\t", [a for (a, b, c) in vector])
        print("\t", [b for (a, b, c) in vector])
        print("\t", [c for (a, b, c) in vector])
