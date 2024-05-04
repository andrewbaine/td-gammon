import itertools
import struct
from typing import List, Tuple

import torch
import torch.nn as nn

from . import epc


def layered(*layers, device=torch.device("cpu")):
    layers = list(
        itertools.chain(
            *[
                (
                    [
                        nn.Linear(n, layers[i + 1], device=device),
                        nn.Sigmoid(),
                    ]
                    if i < (len(layers) - 1)
                    else []
                )
                for (i, n) in enumerate(layers)
            ]
        )
    )
    return nn.Sequential(*layers)


def utility_tensor(device=torch.device("cpu")):
    return torch.tensor([-2, -1, 1, 2], dtype=torch.float, device=device)


def backgammon_utility_tensor(device=torch.device("cpu")):
    return torch.tensor([-3, -2, -1, 1, 2, 3], dtype=torch.float, device=device)


class TurnEncoder(torch.nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super().__init__()
        self.one = torch.tensor([1], dtype=torch.float, device=device)
        self.two = torch.tensor([2], dtype=torch.float, device=device)

    def forward(self, t):
        assert torch.all(torch.logical_or(t == -1, t == 1))
        y = (t + self.one) / self.two
        assert torch.all(torch.logical_or(y == 1, y == 0))
        return y


def barrier_matrix(b, device):
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
        torch.tensor(ms, dtype=torch.float, device=device),
        torch.tensor(scales, dtype=torch.float, device=device),
    )


class Barrier(torch.nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super().__init__()
        self.barrier_additions = torch.tensor(
            [[n - 1 for _ in range(24)] for n in range(1, 8)],
            dtype=torch.float,
            device=device,
        )
        (self.player_1_barrier_matrix, self.barrier_scales_player_1) = barrier_matrix(
            True, device
        )
        (self.player_2_barrier_matrix, self.barrier_scales_player_2) = barrier_matrix(
            False, device
        )
        self.zero24 = torch.tensor(
            [0 for _ in range(24)], dtype=torch.float, device=device
        )
        self.one24 = torch.tensor(
            [1 for _ in range(24)], dtype=torch.float, device=device
        )

    def forward(self, board):
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
    def __init__(self, device=torch.device("cpu")):
        super().__init__(device=device)
        self.zeros_to_the_right = torch.tensor(
            [
                [1 if i == j else -1 if j == i - 1 else 0 for j in range(24)]
                for i in range(24)
            ],
            dtype=torch.float,
            device=device,
        )
        self.zeros_to_the_left = torch.tensor(
            [
                [1 if i == j else -1 if j == i + 1 else 0 for j in range(24)]
                for i in range(24)
            ],
            dtype=torch.float,
            device=device,
        )

    def forward(self, x):
        y = torch.maximum(x, self.zero24)
        r = torch.matmul(y, self.zeros_to_the_left)
        # 3, 2, 1 => 3, 0, 0
        s = torch.where(r >= y, y, self.zero24)

        # -1, -2, -3 => 0, 0, -3
        y2 = torch.minimum(x, self.zero24)
        r2 = torch.matmul(y2, self.zeros_to_the_right)
        s2 = torch.where(r2 <= y2, y2, self.zero24)
        return s + s2


class OneHot(torch.nn.Module):
    def __init__(self, buckets, device):
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
        self.m = torch.tensor(weight, dtype=torch.float, device=device)
        self.b = torch.tensor(bias, dtype=torch.float, device=device)

        self.zeros = torch.zeros_like(self.b)
        self.caps = torch.tensor(caps, dtype=torch.float, device=device)
        self.scale = torch.tensor(scale, dtype=torch.float, device=device).diag()

    def forward(self, x):
        y = torch.matmul(x, self.m) + self.b
        y = torch.maximum(y, self.zeros)
        y = torch.where(y < self.caps, y, self.zeros)
        y = torch.matmul(y, self.scale)
        return y


class TesauroOneHot(OneHot):
    def __init__(self, device=torch.device("cpu")):
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
        super().__init__(buckets, device=device)


class GreatestBarrierBuckets(OneHot):
    def __init__(self, device=torch.device("cpu")):
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
        super().__init__(buckets, device=device)


class Baine(torch.nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super().__init__()
        self.tesauro = TesauroOneHot(device=device)
        self.greatest_barrier = GreatestBarrier(device=device)
        self.greatest_barrier_buckets = GreatestBarrierBuckets(device=device)

    def forward(self, state):
        (_, n) = state.size()
        assert n == 27
        r = self.tesauro(state[:, 0:26])
        s = self.greatest_barrier(state[:, 1:25])
        t = self.greatest_barrier_buckets(s)
        player_bit = state[:, 26:27]
        return torch.cat((r, t, player_bit), dim=1)


class Tesauro(torch.nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super().__init__()
        self.tesauro = TesauroOneHot(device=device)

    def forward(self, state):
        (_, n) = state.size()
        assert n == 27
        r = self.tesauro(state)
        player_bit = state[:, 26:27]
        return torch.cat((r, player_bit), dim=1)


class EPC(torch.nn.Module):
    def __init__(self, db, places, device=torch.device("cpu")):
        super().__init__()
        self.db = db
        self.places = places
        self.device = device

    def f(self, x):
        a = [x if x > 0 else 0 for x in x[0:26]]
        b = [-1 * x if x < 0 else 0 for x in reversed(x[0:26])]

        result = []
        for a in (a, b):
            for s, e in self.places:
                k = epc.make_key(a[s - 1 : e])
                v = self.db.get(k)
                assert v is not None
                (v,) = struct.unpack("f", v)
                result.append(v)
        return result

    def forward(self, x):
        (m, n) = x.size()
        assert m > 0
        assert n == 27
        data = [self.f([int(a) for a in x]) for x in x[:, :26].tolist()]
        return torch.tensor(data, dtype=torch.float, device=self.device)


class HitAvailabilityOneHot(torch.nn.Module):
    def __init__(self, move_vectors, device):
        super().__init__()
        self.move_vectors = move_vectors
        self.device = device

    def forward(self, x: torch.Tensor):
        (m, n) = x.size()
        assert n == 27
        xs = x.unbind(dim=0)
        results = []
        for x in xs:
            x = x.unsqueeze(dim=0)
            assert x.size() == (1, 27)
            result = torch.zeros((4,), device=self.device)
            factor_1 = torch.ones((1,), device=self.device) * 1 / 36
            factor_2 = factor_1 * 2
            zeros = torch.zeros((1,), device=self.device)
            for d1 in range(1, 7):
                for d2 in range(d1, 7):
                    factor = factor_1 if d1 == d2 else factor_2
                    vectors = self.move_vectors.compute_move_vectors(x, (d1, d2))
                    a = torch.min(vectors[:, 0])
                    b = torch.max(vectors[:, 25])
                    c = torch.cat(
                        (
                            torch.where(a < -1, factor, zeros),
                            torch.where(a == -1, factor, zeros),
                            torch.where(b == 1, factor, zeros),
                            torch.where(b > 1, factor, zeros),
                        ),
                        dim=-1,
                    )
                    assert c.size() == (4,)
                    c = c.unsqueeze(dim=0)
                    assert c.size() == (1, 4)
                    result = result + c
            results.append(result)
        print(results)

        results = torch.cat(tuple(results), dim=0)
        print(results.size())
        assert results.size() == (m, 4)
        return results


class BaineEPC(torch.nn.Module):
    def __init__(self, db, places, device) -> None:
        super().__init__()
        self.epc = EPC(db, places, device=device)
        self.baine = Baine(device=device)

    def forward(self, x):
        a = self.baine(x)
        b = self.epc(x)
        return torch.cat((a, b), dim=-1)


class BaineEPCwithHitAvailability(torch.nn.Module):
    def __init__(self, db, places, move_vectors, device) -> None:
        super().__init__()
        self.baine_epc = BaineEPC(db, places, device=device)
        self.hit_availability_one_hot = HitAvailabilityOneHot(
            move_vectors, device=device
        )

    def forward(self, x):
        a = self.baine_epc(x)
        b = self.hit_availability_one_hot(x)
        return torch.cat((a, b), dim=-1)


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
