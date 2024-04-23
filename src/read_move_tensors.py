import bt_2
import torch


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

        noop = bt_2.tensorize(bt_2.noop())
        singles = [bt_2.tensorize(bt_2.all_moves_die(d)) for d in range(1, 7)]

        self.singles_player_1 = singles
        self.singles_player_2 = [for_p2(x) for x in singles]
        self.noop = noop

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
