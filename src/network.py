import torch
import torch.nn as nn

import itertools


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
