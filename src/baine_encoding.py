import torch
from torch import float, logical_and, matmul, maximum, minimum, sub, tensor, where

import tesauro


def tensor(data):
    return torch.tensor(data, dtype=float)


def barrier_matrix(b):
    ms = []

    scales = [
        tensor([(n if b else -n) for _ in range(24)]).diag().tolist()
        for n in range(1, 8)
    ]

    additions = [[n - 1 for _ in range(24)] for n in range(1, 8)]

    for n in range(1, 8):
        row = []
        ms.append(row)
        for i in range(26):
            x = []
            row.append(x)
            for j in range(24):
                x.append(
                    0
                    if (i == 0 or i == 25)
                    else 1 if (-1 < (((i - 1) - j) if b else (j - (i - 1))) < n) else 0
                )
    return (
        torch.tensor(ms, dtype=float),
        torch.tensor(additions, dtype=float),
        torch.tensor(scales, dtype=float),
    )


class Encoder:
    def __init__(self, min=1, max=4):
        assert max >= min
        self.tesauro = tesauro.Encoder()
        (
            self.player_1_barrier_matrix,
            self.barrier_additions,
            self.barrier_scales_player_1,
        ) = barrier_matrix(True)
        self.player_2_barrier_matrix, _, self.barrier_scales_player_2 = barrier_matrix(
            False
        )
        floor = []
        ceil = []
        cap = []
        addition = []
        matrix = []
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
                    addition.append(1 - max)
                    addition.append(1 - max)
                    cap.append(15)
                    cap.append(15)
            assert len(row) == 24 * (max - min + 1) * 2
        self.matrix = torch.tensor(matrix, dtype=torch.float)
        self.floor = torch.tensor(floor, dtype=torch.float)
        self.ceil = torch.tensor(ceil, dtype=torch.float)
        self.addition = torch.tensor(addition, dtype=torch.float)
        self.cap = torch.tensor(cap, dtype=torch.float)

        for x in [
            self.floor,
            self.ceil,
            self.addition,
        ]:
            assert len(x) == 24 * (max - min + 1) * 2
        self.zero_tensor = tensor([0 for _ in range(24 * (max - min + 1) * 2)])
        self.square = torch.diag(tensor([j + 1 for j in range(7)]))
        self.square_neg = torch.neg(self.square)
        self.zeroes = tensor([0 for _ in range(26)])
        self.ones = tensor([1 for _ in range(26)])
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
        self.scale = tensor([1 for _ in range((max - min + 1) * 24 * 2)]).diag()

    def to_(self, device):
        self.tesauro.to_(device)
        self.matrix = self.matrix.to(device=device)
        self.floor = self.floor.to(device=device)
        self.ceil = self.ceil.to(device=device)
        self.addition = self.addition.to(device=device)
        self.cap = self.cap.to(device=device)
        self.zero_tensor = self.zero_tensor.to(device=device)
        self.square = self.square.to(device=device)
        self.square_neg = self.square_neg.to(device=device)
        self.zeroes = self.zeroes.to(device=device)
        self.ones = self.ones.to(device=device)
        self.zeros_to_the_right = self.zeros_to_the_right.to(device=device)
        self.zeros_to_the_left = self.zeros_to_the_left.to(device=device)
        self.zero24 = self.zero24.to(device=device)
        self.scale = self.scale.to(device=device)

        self.player_1_barrier_matrix = self.player_1_barrier_matrix.to(device=device)
        self.player_2_barrier_matrix = self.player_1_barrier_matrix.to(device=device)
        self.barrier_additions = self.barrier_additions.to(device=device)
        self.barrier_scales_player_1 = self.barrier_scales_player_1.to(device=device)
        self.barrier_scales_player_2 = self.barrier_scales_player_2.to(device=device)

    def encode(self, board, player_1):
        y = self.tesauro.encode(board, player_1)
        z = self.encode_step_2(self.encode_step_1(board))
        return torch.cat((y, z), dim=-1)

    def encode_step_1(self, board):
        if board.shape == (26,):
            board = board.unsqueeze(0)
        (n, _) = board.shape

        additions = self.barrier_additions
        scale = self.barrier_scales_player_1
        additions = additions.unsqueeze(1).expand(-1, n, -1)

        points_made = torch.where(board > 1, self.ones, self.zeroes)

        m = self.player_1_barrier_matrix
        a = torch.matmul(points_made, m)
        b = sub(a, additions)
        zero24 = self.zero24.unsqueeze(0).expand(n, -1).unsqueeze(0).expand(7, -1, -1)
        c = maximum(b, zero24)
        d = torch.matmul(c, scale)
        q = torch.max(d, dim=0).values
        r = matmul(q, self.zeros_to_the_left)
        s = where(r == q, r, self.zero24)

        points_made = torch.where(board < -1, self.ones, self.zeroes)
        m = self.player_2_barrier_matrix
        a = torch.matmul(points_made, m)
        b = sub(a, additions)
        c = maximum(b, zero24)
        scale = self.barrier_scales_player_2
        d = torch.matmul(c, scale)
        q = torch.min(d, dim=0).values
        r = matmul(q, self.zeros_to_the_right)
        s2 = where(r == q, r, self.zero24)
        return s + s2

    def encode_step_2(self, board):
        y = matmul(board, self.matrix) + self.addition
        condition = logical_and(self.floor <= y, y < self.ceil)
        y = where(condition, y, self.zero_tensor)
        y = minimum(y, self.cap)
        y = matmul(y, self.scale)
        return y


if __name__ == "__main__":
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
    y = e.encode(board, False)

#    print(board)
#    print("y.size()", y.size())

#    xs = torch.tensor([make_board(), make_board()])
#    ys = e.encode_step_1(xs)
#    print("ys.size()", ys.size())
