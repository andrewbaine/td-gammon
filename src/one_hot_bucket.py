import torch

import torch.nn as nn


class Model(nn.Module):
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
        self.m = torch.tensor(weight, dtype=torch.float32)
        self.b = torch.tensor(bias, dtype=torch.float32)

        self.zeros = torch.zeros_like(self.b)
        self.caps = torch.tensor(caps, dtype=torch.float)
        self.scale = torch.tensor(scale, dtype=torch.float).diag()

    def forward(self, x):
        y = torch.matmul(x, self.m) + self.b
        y = torch.maximum(y, self.zeros)
        y = torch.where(y < self.caps, y, self.zeros)
        y = torch.matmul(y, self.scale)
        return y


from typing import List, Tuple


class Tesauro(Model):
    def __init__(self):
        buckets: List[List[Tuple[int, int, float]]] = [[(-1, -16, 0.5)]]
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


if __name__ == "__main__":

    buckets = [
        [
            (1, 2, 1),
            (-1, -2, 1),
            (2, 3, 1),
            (-2, -3, 1),
            (3, 4, 1),
            (-3, -4, 1),
            (4, 16, 0.5),
            (-4, -16, 0.5),
        ]
        for _ in range(1)
    ]

    model = Model(buckets)
    for i in range(-15, 16, 1):
        print(i, model(torch.tensor([i], dtype=torch.float)).tolist())
