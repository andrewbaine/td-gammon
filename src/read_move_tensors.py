import torch

from write_move_tensors import singles_dir, doubles_dir, ab_dir, noop_dir
import os


def transform(device):
    rows = []
    for i in range(26):
        col = [-1 if ((25 - j) == i) else 0 for j in range(26)]
        rows.append(col)
    return torch.tensor(rows, dtype=torch.int8, device=device)


def for_p2(x, device):
    (moves, low, high, vector) = x
    t = transform(device)
    one = torch.tensor([1 for _ in range(26)], dtype=torch.int8, device=device)
    v = torch.matmul(vector, t)
    h = torch.matmul(low, t).add(one)
    l = torch.matmul(high, t).add(one)
    return (moves, l, h, v)


def read(path, device):
    return tuple(
        torch.load(os.path.join(path, x)).to(device)
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
    def __init__(self, dir, device=torch.device("cuda")):
        self.player_1_vectors = []

        noop = read(noop_dir(dir), device=device)
        singles = [read(singles_dir(dir, d1), device=device) for d1 in range(1, 7)]

        for d1 in range(1, 7):
            for d2 in range(1, d1 + 1):
                if d1 == d2:
                    dubsies = [
                        read(doubles_dir(dir, d1, name), device=device)
                        for name in ["4", "3", "2", "1"]
                    ]
                    dubsies.append(noop)
                    self.player_1_vectors.append(dubsies)
                else:
                    xs = [read(ab_dir(dir, d1, d2), device=device)]
                    xs.append(singles[d1 - 1])
                    xs.append(singles[d2 - 1])
                    xs.append(noop)
                    self.player_1_vectors.append(xs)
        self.player_2_vectors = [
            [for_p2(y, device) for y in x] for x in self.player_1_vectors
        ]

    def compute_moves(self, state):
        (board, player_1, (d1, d2)) = state
        i = find_index(d1, d2)
        for moves, lower, upper, vector in (
            self.player_1_vectors if player_1 else self.player_2_vectors
        )[i]:
            indices = torch.all(lower <= board, dim=1) & torch.all(upper > board, dim=1)
            if torch.numel(vector[indices]) > 0:
                return moves[indices]
        assert False

    def compute_move_vectors(self, state):
        (board, player_1, (d1, d2)) = state
        i = find_index(d1, d2)
        for moves, lower, upper, vector in (
            self.player_1_vectors if player_1 else self.player_2_vectors
        )[i]:
            indices = torch.all(lower <= board, dim=1) & torch.all(upper > board, dim=1)
            if torch.numel(vector[indices]) > 0:
                return vector[indices]
        assert False
