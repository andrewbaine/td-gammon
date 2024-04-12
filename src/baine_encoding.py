import torch

from torch import matmul, maximum, minimum, logical_and, float, where


def tensor(data):
    return torch.tensor(data, dtype=float)


def barrier_matrix(b):
    m = []
    for n in range(1, 8):
        x = []
        m.append(x)
        for i in range(26):
            row = []
            x.append(row)
            for j in range(24):
                row.append(
                    0
                    if (i == 0 or i == 25)
                    else 1 if (-1 < (((i - 1) - j) if b else (j - (i - 1))) < n) else 0
                )
    return tensor(m)


def additions():
    m = []
    for n in range(1, 8):
        m.append([(n - 1) for _ in range(24)])
    return tensor(m)


class Encoder:
    def __init__(self, min=1, max=4):
        assert max >= min
        self.player_1_barrier_matrix = barrier_matrix(True)
        self.player_2_barrier_matrix = barrier_matrix(False)
        self.additions = additions()
        matrix = []
        floor = []
        ceil = []
        cap = []
        addition = []
        floor_2 = []
        ceil_2 = []
        for i in range(24):
            row = []
            matrix.append(row)
            for j in range(24):
                for k in range(min, max):
                    row.append(1 if i == j else 0)
                    row.append(-1 if i == j else 0)
                    if i == 0:
                        floor.append(k)
                        ceil.append(k + 1)
                        floor.append(k)
                        ceil.append(k + 1)
                        floor_2.append(0)
                        ceil_2.append(1)
                        floor_2.append(0)
                        ceil_2.append(1)
                        addition.append(0)
                        addition.append(0)
                        cap.append(1)  # unary encoding for x = 1, 2, or 3
                        cap.append(1)  # unary encoding for x = -1, -2, or -3
                # this is for the excess
                row.append(1 if i == j else 0)
                row.append(-1 if i == j else 0)
                if i == 0:
                    floor.append(0)
                    ceil.append(16)
                    floor.append(0)
                    ceil.append(16)
                    floor_2.append(0)
                    ceil_2.append(16)
                    floor_2.append(0)
                    ceil_2.append(16)
                    addition.append(-max)
                    addition.append(-max)
                    cap.append(15)
                    cap.append(15)
            assert len(row) == 24 * (max - min + 1) * 2
        self.matrix = torch.tensor(matrix, dtype=torch.float)
        self.floor = torch.tensor(floor, dtype=torch.float)
        self.ceil = torch.tensor(ceil, dtype=torch.float)
        self.floor_2 = torch.tensor(floor_2, dtype=torch.float)
        self.ceil_2 = torch.tensor(ceil_2, dtype=torch.float)
        self.addition = torch.tensor(addition, dtype=torch.float)
        self.cap = torch.tensor(cap, dtype=torch.float)

        for x in [
            self.floor,
            self.ceil,
            self.floor_2,
            self.ceil_2,
            self.addition,
        ]:
            assert len(x) == 24 * (max - min + 1) * 2
        self.zero_tensor = tensor([0 for _ in range(24 * (max - min + 1) * 2)])
        self.square = torch.diag(tensor([j + 1 for j in range(7)]))
        self.square_neg = torch.neg(self.square)
        self.zeroes = tensor([0 for _ in range(26)])
        self.ones = tensor([1 for _ in range(26)])
        self.many_zeroes = torch.matmul(self.zeroes, self.player_1_barrier_matrix)
        self.zeros_to_the_right = tensor(
            [
                [1 if i == j else -1 if j == i - 1 else 0 for j in range(24)]
                for i in range(24)
            ]
        )
        self.zeros_to_the_left = tensor(
            [
                [1 if i == j else -1 if j == i + 1 else 0 for j in range(24)]
                for i in range(24)
            ]
        )
        self.zero24 = tensor([0 for _ in range(24)])

    def encode(self, board, player_1):
        m = self.player_1_barrier_matrix
        y = torch.matmul(torch.where(board > 1, self.ones, self.zeroes), m)
        z = y - self.additions
        p = maximum(z, self.many_zeroes)
        q = matmul(self.square, p).max(dim=0).values
        r = matmul(q, self.zeros_to_the_left)
        s = where(r == q, r, self.zero24)

        m2 = self.player_2_barrier_matrix
        y2 = matmul(torch.where(board < -1, self.ones, self.zeroes), m2)
        z2 = y2 - self.additions
        p2 = maximum(z2, self.many_zeroes)
        q2 = matmul(self.square_neg, p2).min(dim=0).values
        r2 = matmul(
            q2,
            self.zeros_to_the_right,
        )
        s2 = where(r2 == q2, r2, self.zero24)
        return s + s2

    def encode_step_2(self, board, player_1):
        y = matmul(board, self.matrix) + self.addition
        condition = logical_and(self.floor <= y, y < self.ceil)
        y = where(condition, y, self.zero_tensor)
        y = minimum(y, self.cap)
        y = matmul(y, self.scale)


e = Encoder()
board = tensor(
    [
        5,
        -2,
        -2,
        -2,
        -1,
        2,
        2,
        0,
        0,
        1,
        2,
        -2,
        2,
        2,
        -2,
        -2,
        -3,
        -1,
        -2,
        3,
        3,
        3,
        0,
        0,
        0,
        0,
    ]
)
print(board)
print(e.encode(board, False))
