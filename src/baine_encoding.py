import torch

from torch import matmul, maximum, minimum, float


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
            for j in range(26):
                row.append(
                    0
                    if (i == 0 or i == 25 or j == 0 or j == 25)
                    else 1 if (-1 < ((i - j) if b else (j - i)) < n) else 0
                )
    return tensor(m)


def additions():
    m = []
    for n in range(1, 8):
        m.append([(n - 1) for _ in range(26)])
    return tensor(m)


class Encoder:
    def __init__(self):
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
        for i in range(26):
            row = []
            matrix.append(row)
            row.append(-1 if i == 0 else 0)  # player 2 bar
            if i == 0:
                floor.append(0)
                ceil.append(16)
                floor_2.append(-15)
                ceil_2.append(0)
                addition.append(0)
                cap.append(15)
            for j in range(1, 25):
                for k in range(0, 3):
                    row.append(1 if i == j else 0)
                    row.append(-1 if i == j else 0)
                    if i == 0:
                        floor.append(k + 1)
                        ceil.append(k + 2)
                        floor.append(k + 1)
                        ceil.append(k + 2)
                        floor_2.append(0)
                        ceil_2.append(1)
                        floor_2.append(0)
                        ceil_2.append(1)
                        addition.append(0)
                        addition.append(0)
                        cap.append(1)  # unary encoding for x = 1, 2, or 3
                        cap.append(1)  # unary encoding for x = -1, -2, or -3
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
                    addition.append(-3)
                    addition.append(-3)
                    cap.append(15)
                    cap.append(15)
            row.append(1 if i == 25 else 0)  # player 1 bar
            if i == 25:
                floor.append(0)
                ceil.append(16)
                floor_2.append(0)
                ceil_2.append(15)
                addition.append(0)
                cap.append(15)
            for _ in range(4):
                row.append(0)
            assert len(row) == 198
        for x in [floor, ceil, floor_2, ceil_2, addition]:
            for _ in range(4):
                x.append(0)
        for i in range(4):
            cap.append(15)
        self.matrix = torch.tensor(matrix, dtype=torch.float)
        self.floor = torch.tensor(floor, dtype=torch.float)
        self.ceil = torch.tensor(ceil, dtype=torch.float)
        self.floor_2 = torch.tensor(floor_2, dtype=torch.float)
        self.ceil_2 = torch.tensor(ceil_2, dtype=torch.float)
        self.addition = torch.tensor(addition, dtype=torch.float)
        self.cap = torch.tensor(cap, dtype=torch.float)
        count_white = [[1 if i == 194 else 0 for i in range(198)] for _ in range(26)]
        count_black = [[-1 if i == 195 else 0 for i in range(198)] for _ in range(26)]
        self.count_white_pieces = torch.tensor(count_white, dtype=torch.float)
        self.count_black_pieces = torch.tensor(count_black, dtype=torch.float)
        self.zero_board = torch.tensor([0 for _ in range(26)], dtype=torch.float)
        self.white_turn = torch.tensor(
            [(1 if i == 196 else 0) for i in range(198)],
            dtype=torch.float,
        )
        self.black_turn = torch.tensor(
            [(1 if i == 197 else 0) for i in range(198)],
            dtype=torch.float,
        )
        for x in [
            self.floor,
            self.ceil,
            self.white_turn,
            self.black_turn,
            self.floor_2,
            self.ceil_2,
            self.addition,
        ]:
            assert len(x) == 198
        self.zero_tensor = torch.tensor(
            [0 for _ in range(198)],
            dtype=torch.float,
        )
        scale = []
        for i in range(198):
            row = []
            scale.append(row)
            for j in range(198):
                if i == j:
                    if i == 0:
                        row.append(0.5)
                    elif 0 < i < 8 * 24 + 1:
                        m = i % 8
                        if m == 0 or m == 7:
                            row.append(0.5)
                        else:
                            row.append(1)
                    elif i == 8 * 24 + 1:
                        row.append(0.5)
                    elif i == 8 * 24 + 2 or i == 8 * 24 + 3:
                        row.append(1.0 / 15.0)
                    elif i == 8 * 24 + 4 or i == 8 * 24 + 5:
                        row.append(1.0)
                    else:
                        print(i)
                        assert False
                else:
                    row.append(0)
        self.scale = torch.tensor(
            scale,
            dtype=torch.float,
        )
        self.square = tensor(
            [[j + 1 if i == j else 0 for j in range(7)] for i in range(7)]
        )
        self.square_neg = tensor(
            [[-j - 1 if i == j else 0 for j in range(7)] for i in range(7)]
        )
        self.zeroes = tensor([0 for _ in range(26)])
        self.ones = tensor([1 for _ in range(26)])
        self.many_zeroes = torch.matmul(self.zeroes, self.player_1_barrier_matrix)

    def encode(self, board, player_1):
        m = self.player_1_barrier_matrix
        y = torch.matmul(torch.where(board > 1, self.ones, self.zeroes), m)
        z = y - self.additions
        p = torch.maximum(z, self.many_zeroes)
        q = torch.matmul(self.square, p).max(dim=0).values
        m2 = self.player_2_barrier_matrix
        y2 = torch.matmul(torch.where(board < -1, self.ones, self.zeroes), m2)
        z2 = y2 - self.additions
        p2 = torch.maximum(z2, self.many_zeroes)
        q2 = torch.matmul(self.square_neg, p2).min(dim=0).values
        return q + q2
