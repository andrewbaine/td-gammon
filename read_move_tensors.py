import torch

from write_move_tensors import singles_dir, doubles_dir, ab_dir, noop_dir
import os


def read(path):
    return tuple(
        torch.load(os.path.join(path, x))
        for x in ["moves.pt", "low.pt", "high.pt", "vector.pt"]
    )


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
        case 4:  # 3, 4, 5
            return 2 + b
        case 5:  # 6, 7, 8, 9
            return 5 + b
        case 6:  # 10, 11, 12, 13, 14, 15
            return 9 + b
        case _:
            assert False


class MoveTensors:
    def __init__(self, dir, device="cpu"):
        self.doubles = []
        self.singles = []
        self.ab = []

        with torch.device(device):
            self.noop = read(noop_dir(dir))
            for d1 in range(1, 7):
                self.singles.append(read(singles_dir(dir, d1)))
                dubsies = [
                    read(doubles_dir(dir, d1, name)) for name in ["4", "3", "2", "1"]
                ]
                dubsies.append(self.noop)
                self.doubles.append(dubsies)
                for d2 in range(1, d1):
                    self.ab.append(read(ab_dir(dir, d1, d2)))

    def compute_moves(self, state):
        (board, player_1, (d1, d2)) = state
        assert player_1
        if d1 == d2:
            dubsies = self.doubles[d1 - 1]
            for moves, lower, upper, vector in dubsies:
                indices = torch.all(lower <= board, dim=1) & torch.all(
                    upper > board, dim=1
                )
                ms = moves[indices]
                if ms.size()[0] > 0:
                    return ms
        else:
            i = find_index(d1, d2)
            big, small = (d1, d2) if d1 > d2 else (d2, d1)
            for moves, lower, upper, vector in [
                self.ab[i],
                self.singles[big - 1],
                self.singles[small - 1],
                self.noop,
            ]:
                indices = torch.all(lower <= board, dim=1) & torch.all(
                    upper > board, dim=1
                )
                ms = moves[indices]
                if ms.size()[0] > 0:
                    return ms
        assert False
