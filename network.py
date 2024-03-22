import torch
import torch.nn as nn

from collections import OrderedDict


def layered(*layers):
    return nn.Sequential()


def utility():
    u = nn.Linear(4, 1, bias=False)
    u.weight = nn.Parameter(
        torch.tensor([x for x in (-2, -1, 1, 2)], dtype=torch.float),
    )
    return u


def with_utility(network):
    return nn.Sequential(OrderedDict([("network", network), ("utility", utility())]))
