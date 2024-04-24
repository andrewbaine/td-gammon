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


def impossible_hit(start, end):
    return ([start, end, 1], [0 for _ in r()], [0 for _ in r()], [0 for _ in r()])


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
        return [([start, end, 0], low, high, vector), impossible_hit(start, end)]
    elif end == 0:
        # exact bearoff
        low, high, vector = split_out(
            [empty_t if i > 6 else start_t if i == start else any_t for i in r()]
        )
        return [([start, end, 0], low, high, vector), impossible_hit(start, end)]
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
        if low < high:
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


def transform():
    rows = []
    for i in range(26):
        col = [-1 if ((25 - j) == i) else 0 for j in range(26)]
        rows.append(col)
    return torch.tensor(rows, dtype=torch.float)


def for_p2(x):
    (moves, low, high, vector) = x
    t = transform()
    one = torch.tensor([1 for _ in range(26)], dtype=torch.float)
    v = torch.matmul(vector, t)
    h = torch.matmul(low, t).add(one)
    l = torch.matmul(high, t).add(one)
    return (moves, l, h, v)


class MoveTensors:
    def __init__(self):

        singles = [tensorize(all_moves_die(d)) for d in range(1, 7)]

        self.singles_player_1 = singles
        self.singles_player_2 = [for_p2(x) for x in singles]
        self.noop = tensorize(noop())

    def to_(self, device):
        self.singles_player_1 = [
            (
                m.to(device=device),
                l.to(device=device),
                u.to(device=device),
                v.to(device=device),
            )
            for (m, l, u, v) in self.singles_player_1
        ]
        self.singles_player_2 = [
            (
                m.to(device=device),
                l.to(device=device),
                u.to(device=device),
                v.to(device=device),
            )
            for (m, l, u, v) in self.singles_player_2
        ]
        (m, l, u, v) = self.noop
        self.noop = (
            m.to(device=device),
            l.to(device=device),
            u.to(device=device),
            v.to(device=device),
        )

    def movesies(self, board, xs, short_circuit=True):
        (_, _, _, move_vectors) = self.noop
        (m, n) = board.size()
        for _, lower, upper, vector in xs:
            m = vector.size()[0]
            n = move_vectors.size()[0]
            mv = move_vectors.unsqueeze(0).expand(m, -1, -1)
            b = board.unsqueeze(0).expand(m, -1, -1)
            (lower, upper, vector) = (
                lower.unsqueeze(1).expand(-1, n, -1),
                upper.unsqueeze(1).expand(-1, n, -1),
                vector.unsqueeze(1).expand(-1, n, -1),
            )
            indices = torch.logical_and(lower <= board, upper > board)
            indices = torch.all(indices, dim=-1)
            (v, b, mv) = torch.stack((vector, b, mv), -1)[indices].unbind(-1)

            if v.numel() == 0 and short_circuit:
                return move_vectors
            board = b + v
            move_vectors = mv + v
        return move_vectors

    def dubsies(self, board, player_1, d):
        x = (self.singles_player_1 if player_1 else self.singles_player_2)[d - 1]
        return self.movesies(board, [x, x, x, x], short_circuit=True)

    def compute_move_vectors_old(
        self,
        state,
    ):
        (board, player_1, (d1, d2)) = state
        if d1 == d2:
            return self.dubsies(board, player_1, d1)

        v = self.singles_player_1 if player_1 else self.singles_player_2
        (x1, x2) = (v[d1 - 1], v[d2 - 1]) if d1 > d2 else (v[d2 - 1], v[d1 - 1])

        a = self.movesies(board, [x1, x2], short_circuit=False)
        b = self.movesies(board, [x2, x1], short_circuit=False)
        if a.numel() > 0 or b.numel() > 0:
            catted = torch.cat((a, b))
            unique = torch.unique(catted, dim=0)
            return unique
        c = self.movesies(board, [x1], short_circuit=False)
        if c.numel() > 0:
            return c
        d = self.movesies(board, [x2], short_circuit=False)
        if d.numel() > 0:
            return d
        (_, _, _, move_vectors) = self.noop
        return move_vectors

    def compute_move_vectors(self, board, dice):
        player_1 = board[0][26]
        assert board.size() == (1, 27), board.size()
        vs = self.compute_move_vectors_old((board[:, :26], player_1, dice))
        (n, m) = vs.size()
        assert m == 26
        next_player = (1 - 2 * player_1).unsqueeze(dim=0).expand(n, 1)
        result = torch.cat((vs, next_player), dim=1)
        assert result.size() == (n, 27)
        return result
