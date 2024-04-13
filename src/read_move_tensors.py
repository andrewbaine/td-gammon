import torch

from write_move_tensors import singles_dir, doubles_dir, ab_dir, noop_dir
import os


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


def read(path):
    return tuple(
        torch.load(os.path.join(path, x)).to(dtype=torch.float)
        for x in ["moves.pt", "low.pt", "high.pt", "vector.pt"]
    )


def find_index(a, b):
    if a < b:
        (a, b) = (b, a)
    assert a >= b
    match a:
        case 1:
            assert b == 1
            return b - 1
        case 2:  # 0
            return b
        case 3:  # 1, 2
            return b + 2
        case 4:  # 3, 4, 5
            return b + 5
        case 5:  # 6, 7, 8, 9
            return b + 9
        case 6:  # 10, 11, 12, 13, 14, 15
            return b + 14
        case _:
            assert False


class MoveTensors:
    def __init__(self, dir):
        self.player_1_vectors = []

        noop = read(noop_dir(dir))
        singles = [read(singles_dir(dir, d1)) for d1 in range(1, 7)]

        for d1 in range(1, 7):
            for d2 in range(1, d1 + 1):
                if d1 == d2:
                    dubsies = [
                        read(doubles_dir(dir, d1, name))
                        for name in ["4", "3", "2", "1"]
                    ]
                    dubsies.append(noop)
                    self.player_1_vectors.append(dubsies)
                else:
                    xs = [read(ab_dir(dir, d1, d2))]
                    xs.append(singles[d1 - 1])
                    xs.append(singles[d2 - 1])
                    xs.append(noop)
                    self.player_1_vectors.append(xs)
        self.player_2_vectors = [[for_p2(y) for y in x] for x in self.player_1_vectors]
        self.singles_player_1 = singles
        self.singles_player_2 = [for_p2(x) for x in singles]

    def to_(self, device):
        self.player_1_vectors = [
            [
                (
                    m.to(device=device),
                    l.to(device=device),
                    u.to(device=device),
                    v.to(device=device),
                )
                for (m, l, u, v) in x
            ]
            for x in self.player_1_vectors
        ]
        self.player_2_vectors = [
            [
                (
                    m.to(device=device),
                    l.to(device=device),
                    u.to(device=device),
                    v.to(device=device),
                )
                for (m, l, u, v) in x
            ]
            for x in self.player_2_vectors
        ]
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

    def compute_moves(self, state):
        (board, player_1, (d1, d2)) = state
        i = find_index(d1, d2)
        for moves, lower, upper, vector in (
            self.player_1_vectors if player_1 else self.player_2_vectors
        )[i]:
            indices = torch.all(lower <= board, dim=1) & torch.all(upper > board, dim=1)
            if torch.numel(vector[indices]) > 0:
                return (moves[indices], vector[indices])
        assert False

    def compute_move_vectors_v1(self, state):
        (board, player_1, (d1, d2)) = state
        i = find_index(d1, d2)
        for moves, lower, upper, vector in (
            self.player_1_vectors if player_1 else self.player_2_vectors
        )[i]:
            indices = torch.all(torch.logical_and(lower <= board, upper > board), dim=1)
            vi = vector[indices]
            if torch.numel(vi) > 0:
                return vi
        assert False

    def dubsies(self, board, player_1, d):
        raise Exception("were not there yet")
        pass

    def compute_move_vectors_2(self, state):
        (board, player_1, (d1, d2)) = state
        if d1 == d2:
            return self.dubsies(board, player_1, d1)

        v = self.singles_player_1 if player_1 else self.singles_player_2

        singletons = []
        doubletons = []
        for d1, d2 in [(d1, d2), (d2, d1)] if d1 > d2 else [(d2, d1), (d1, d2)]:
            (moves, lower, upper, vector) = v[d1 - 1]
            indices = torch.all(torch.logical_and(lower <= board, upper > board), dim=1)

            vector_moves = vector[indices]
            board_after_d1 = vector[indices] + board  # n x 26
            n = board_after_d1.size()
            assert n[1] == 26
            n = n[0]
            (moves, lower, upper, vector) = v[d2 - 1]  # m x 26
            m = vector.size()
            assert m[1] == 26
            m = m[0]

            board_after_d1 = board_after_d1.unsqueeze(0).expand(m, -1, -1)
            vector_moves = vector_moves.unsqueeze(0).expand(m, -1, -1)

            (moves, lower, upper, vector) = (
                moves.unsqueeze(1).expand(-1, n, -1),
                lower.unsqueeze(1).expand(-1, n, -1),
                upper.unsqueeze(1).expand(-1, n, -1),
                vector.unsqueeze(1).expand(-1, n, -1),
            )
            assert lower.size() == board_after_d1.size()
            indices = torch.logical_and(lower <= board_after_d1, upper > board_after_d1)
            indices = torch.all(indices, dim=-1)
            y = vector_moves[indices] + vector[indices]
            doubletons.append(y)
        catted = torch.cat((doubletons[0], doubletons[1]))
        unique = torch.unique(catted, dim=0)

        return unique

    def compute_move_vectors(self, state):
        return self.compute_move_vectors_2(state)
