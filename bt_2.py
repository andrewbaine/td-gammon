start_t = (1, 16, -1)
move_t = (0, 16, 1)
hit_t = (-1, 0, 2)
bar_t = (-15, 1, -1)
empty_t = (-15, 1, 0)
any_t = (-15, 16, 0)


def memoize(f):
    cache = {}

    def g(limit, dice):
        key = (limit, tuple(dice))
        if key not in cache:
            cache[key] = f(limit, dice)
        return cache[key]

    return g


def r():
    return range(26)


def moves(start, die):
    end = start - die
    move = (start, end)
    if end < 0:
        # over-bearoff
        move = [empty_t if i > start else start_t if i == start else any_t for i in r()]
        return [([(start, end, False)], move)]
    elif end == 0:
        # exact bearoff
        move = [empty_t if i > 6 else start_t if i == start else any_t for i in r()]
        return [([(start, end, False)], move)]
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
            return [([(start, end, False)], move), ([(start, end, True)], hit)]
        assert start != 25
        move = [
            (
                empty_t
                if i == 25 or i == end
                else start_t if i == start else move_t if i == end else any_t
            )
            for i in r()
        ]
        hit = [
            (
                empty_t
                if i == 25
                else (
                    start_t
                    if i == start
                    else hit_t if i == end else bar_t if i == 0 else any_t
                )
            )
            for i in r()
        ]
        return [([(start, end, False)], move), ([(start, end, True)], hit)]


for x in moves(25, 6):
    print(x)
