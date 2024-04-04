import torch

from write_move_tensors import singles_dir, doubles_dir, ab_dir, noop_dir
import os


def transform():
    rows = []
    for i in range(26):
        col = [0 if ((25 - j) == i) else -1 for j in range(26)]
        rows.append(col)
    return rows


def read(path):
    return tuple(
        torch.load(os.path.join(path, x))
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
    def __init__(self, dir, device="cpu"):
        self.ab = []

        with torch.device(device):
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
                        self.ab.append(dubsies)
                    else:
                        xs = [read(ab_dir(dir, d1, d2))]
                        xs.append(singles[d1 - 1])
                        xs.append(singles[d2 - 1])
                        xs.append(noop)
                        self.ab.append(xs)

    def compute_moves(self, state):
        (board, player_1, (d1, d2)) = state
        assert player_1
        i = find_index(d1, d2)
        tensors = self.ab[i]
        for moves, lower, upper, vector in tensors:
            indices = torch.all(lower <= board, dim=1) & torch.all(upper > board, dim=1)
            ms = moves[indices]
            if ms.size()[0] > 0:
                return ms
        assert False
