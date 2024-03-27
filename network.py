import torch
import torch.nn as nn

import itertools
from collections import OrderedDict


def layered(*layers, softmax=False):
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
    if softmax:
        layers.append(nn.Softmax(dim=0))
    return nn.Sequential(*layers)


def utility_tensor():
    return torch.tensor([x for x in (-2, -1, 1, 2)], dtype=torch.float)


def utility():
    u = nn.Linear(4, 1, bias=False)
    u.weight = nn.Parameter(
        utility_tensor(),
    )
    return u


def with_utility(network):
    return nn.Sequential(OrderedDict([("network", network), ("utility", utility())]))
