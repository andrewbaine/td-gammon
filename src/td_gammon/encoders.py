import itertools
from typing import List, Tuple

import torch
import torch
import torch.nn as nn


def tensor(data):
    return torch.tensor(data, dtype=torch.float, requires_grad=False)


def layered(*layers):
    layers = list(
        itertools.chain(
            *[
                (
                    [nn.Linear(n, layers[i + 1]), nn.Sigmoid()]
                    if i < (len(layers) - 1)
                    else []
                )
                for (i, n) in enumerate(layers)
            ]
        )
    )
    return nn.Sequential(*layers)


def utility_tensor():
    return torch.tensor([-2, -1, 1, 2], dtype=torch.float)


def backgammon_utility_tensor():
    return torch.tensor([-3, -2, -1, 1, 2, 3], dtype=torch.float)


class TurnEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.one = tensor([1])
        self.two = tensor([2])

    def forward(self, t):
        assert torch.all(torch.logical_or(t == -1, t == 1))
        y = (t + self.one) / self.two
        assert torch.all(torch.logical_or(y == 1, y == 0))
        return y


def barrier_matrix(b):
    ms = []

    scales = [
        torch.tensor([(n if b else -n) for _ in range(24)]).diag().tolist()
        for n in range(1, 8)
    ]

    for n in range(1, 8):
        row = []
        ms.append(row)
        for i in range(24):
            x = []
            row.append(x)
            for j in range(24):
                x.append(1 if (-1 < ((i - j) if b else (j - i)) < n) else 0)
    return (
        torch.tensor(ms, dtype=torch.float),
        torch.tensor(scales, dtype=torch.float),
    )


class Barrier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.barrier_additions = torch.tensor(
            [[n - 1 for _ in range(24)] for n in range(1, 8)], dtype=torch.float
        )
        (self.player_1_barrier_matrix, self.barrier_scales_player_1) = barrier_matrix(
            True
        )
        (self.player_2_barrier_matrix, self.barrier_scales_player_2) = barrier_matrix(
            False
        )
        self.zero24 = torch.tensor([0 for _ in range(24)], dtype=torch.float)
        self.one24 = torch.tensor([1 for _ in range(24)], dtype=torch.float)

    def forward(self, board):
        #        if board.shape == (26,):
        #            board = board.unsqueeze(0)
        (n, _) = board.shape

        additions = self.barrier_additions
        additions = additions.unsqueeze(1).expand(-1, n, -1)

        points_made = torch.where(board > 1, self.one24, self.zero24)

        m = self.player_1_barrier_matrix
        a = torch.matmul(points_made, m)
        b = torch.sub(a, additions)
        zero24 = self.zero24.unsqueeze(0).expand(n, -1).unsqueeze(0).expand(7, -1, -1)
        c = torch.maximum(b, zero24)
        d = torch.matmul(c, self.barrier_scales_player_1)
        q1 = torch.max(d, dim=0).values

        points_made = torch.where(board < -1, self.one24, self.zero24)
        m = self.player_2_barrier_matrix
        a = torch.matmul(points_made, m)
        b = torch.sub(a, additions)
        c = torch.maximum(b, zero24)
        d = torch.matmul(c, self.barrier_scales_player_2)
        q2 = torch.min(d, dim=0).values

        return q1 + q2


class GreatestBarrier(Barrier):
    def __init__(self):
        super().__init__()
        self.zeros_to_the_right = torch.tensor(
            [
                [1 if i == j else -1 if j == i - 1 else 0 for j in range(24)]
                for i in range(24)
            ],
            dtype=torch.float,
        )
        self.zeros_to_the_left = torch.tensor(
            [
                [1 if i == j else -1 if j == i + 1 else 0 for j in range(24)]
                for i in range(24)
            ],
            dtype=torch.float,
        )

    def forward(self, x):
        y = torch.maximum(x, self.zero24)
        r = torch.matmul(y, self.zeros_to_the_left)
        # 3, 2, 1 => 3, 0, 0
        s = torch.where(r >= y, y, self.zero24)

        #
        y2 = torch.minimum(x, self.zero24)
        r2 = torch.matmul(y2, self.zeros_to_the_right)
        s2 = torch.where(r2 <= y2, y2, self.zero24)
        return s + s2


class OneHot(torch.nn.Module):
    def __init__(self, buckets):
        super().__init__()
        weight = []
        bias = []
        caps = []
        scale = []
        for i in range(len(buckets)):
            row = []
            weight.append(row)
            for j, bucket in enumerate(buckets):
                for a, b, s in bucket:
                    row.append((1 if a > 0 else -1) if i == j else 0)
                    if i == 0:
                        scale.append(s)
                        bias.append((1 - a) if a > 0 else (a + 1))
                        caps.append(1 + abs(b - a))
        self.m = tensor(weight)
        self.b = tensor(bias)

        self.zeros = torch.zeros_like(self.b)
        self.caps = tensor(caps)
        self.scale = tensor(scale).diag()

    def forward(self, x):
        y = torch.matmul(x, self.m) + self.b
        y = torch.maximum(y, self.zeros)
        y = torch.where(y < self.caps, y, self.zeros)
        y = torch.matmul(y, self.scale)
        return y


class TesauroOneHot(OneHot):
    def __init__(self):
        buckets: List[List[Tuple[int, int, float]]] = []
        buckets.append([(-1, -16, 0.5)])
        for _ in range(24):
            buckets.append(
                [
                    (1, 2, 1.0),
                    (-1, -2, 1.0),
                    (2, 3, 1.0),
                    (-2, -3, 1.0),
                    (3, 4, 1.0),
                    (-3, -4, 1.0),
                    (4, 16, 0.5),
                    (-4, -16, 0.5),
                ]
            )
        buckets.append([(1, 16, 1.0 / 0.5)])
        super().__init__(buckets)


class GreatestBarrierBuckets(OneHot):
    def __init__(self):
        buckets = []
        for i in range(0, 24):
            bucket = []
            buckets.append(bucket)
            bucket.append((1, 2, 1.0))
            bucket.append((-1, -2, 1.0))
            if i < (24 - 1):
                bucket.append((2, 3, 1.0))
            if i > 0:
                bucket.append((-2, -3, 1.0))
            if i < (24 - 2):
                bucket.append((3, 8, 0.5))
            if i > 1:
                bucket.append((-3, -8, 0.5))
        super().__init__(buckets)


class Baine(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tesauro = TesauroOneHot()
        self.greatest_barrier = GreatestBarrier()
        self.greatest_barrier_buckets = GreatestBarrierBuckets()

    def forward(self, state):
        (_, n) = state.size()
        assert n == 27
        r = self.tesauro(state[:, 0:26])
        s = self.greatest_barrier(state[:, 1:25])
        t = self.greatest_barrier_buckets(s)
        player_bit = state[:, 26:27]
        return torch.cat((r, t, player_bit), dim=1)


class Tesauro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tesauro = TesauroOneHot()

    def forward(self, state):
        (_, n) = state.size()
        assert n == 27
        r = self.tesauro(state)
        player_bit = state[:, 26:27]
        return torch.cat((r, player_bit), dim=1)


class Evaluator(torch.nn.Module):
    def __init__(self, encoder, network, utility):
        super().__init__()
        self.encoder = encoder
        self.network = network
        assert utility.size() == (4,) or utility.size() == (6,)
        self.utility = utility.unsqueeze(1)

    def forward(self, x):
        (m, n) = x.size()
        assert m > 0
        assert n == 27
        a = self.encoder(x)
        b = self.network(a)
        assert b.size() == (m, self.utility.size()[0])
        c = torch.softmax(b, dim=1)
        assert c.size() == (m, self.utility.size()[0])
        d = torch.matmul(c, self.utility)
        assert d.size() == (m, 1)
        return d
