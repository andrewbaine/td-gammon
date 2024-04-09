import torch
import torch.nn as nn

import itertools
from collections import OrderedDict


def layered(*layers, softmax=False, device=torch.device("cuda")):
    layers = list(
        itertools.chain(
            *[
                (
                    [nn.Linear(n, layers[i + 1], device=device), nn.Sigmoid()]
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


def utility_tensor(device=torch.device("cuda")):
    return torch.tensor(
        [-2, -1, 1, 2], dtype=torch.float, requires_grad=False, device=device
    )


def utility(device=torch.device("cuda")):
    u = nn.Linear(4, 1, bias=False, dtype=torch.float, device=device)
    u.weight = nn.Parameter(utility_tensor(), requires_grad=False)
    u.to(device=device)
    return u


def with_utility(network, device=torch.device("cuda")):
    return nn.Sequential(
        OrderedDict([("network", network), ("utility", utility(device=device))])
    )


def backgammon_utility_tensor(device=torch.device("cuda")):
    return torch.tensor([-3, -2, -1, 1, 2, 3], dtype=torch.float, device=device)


def backgammon_utility(device=torch.device("cuda")):
    u = nn.Linear(6, 1, bias=False, dtype=float)
    u.weight = nn.Parameter(backgammon_utility_tensor(device=device))
    return u


def with_backgammon_utility(network, device):
    return nn.Sequential(network, backgammon_utility(device=device))
