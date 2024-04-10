import torch
import network
import pytest

import torch
import network

net = torch.nn.Linear(4, 1, bias=False)
net.weight = torch.nn.Parameter(network.utility_tensor())


cases = [
    ([1, 2, 3, 4], 7),
    ([1, 3, 5, 7], 14),
    ([1, 2, 4, 8], 16),
    ([4, 3, 2, 1], -7),
]


@pytest.mark.parametrize("t", cases)
def test(t):
    (input, output) = t
    v = net(torch.tensor(input, dtype=torch.float)).item()
    assert v == output
