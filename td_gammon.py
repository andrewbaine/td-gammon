import torch.nn as nn
import itertools

import torch

utility_tensor = (2, 1, -1, -2)


class Network(nn.Sequential):
    # http://incompleteideas.net/book/first/ebook/node87.html
    def __init__(self, *layers):
        super().__init__(
            *list(
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
        )


def observe(gamestate, tensor):
    (board, player_1) = gamestate
    a = 15.0
    b = 15.0
    for i in range(0, 24):
        for j in range(0, 3):
            tensor[4 * i + j] = 0.0
            tensor[98 + 4 * i + j] = 0.0
        pc = board[i + 1]
        if pc > 0:
            a -= pc
            tensor[4 * i] = 1.0
            if pc > 1:
                tensor[4 * i + 1] = 1.0
                if pc > 2:
                    tensor[4 * i + 2] = 1.0
                    if pc > 3:
                        tensor[4 * i + 3] = (pc - 3.0) / 2.0
        elif pc < 0:
            b += pc
            tensor[98 + 4 * i] = 1.0
            if pc < -1:
                tensor[98 + 4 * i + 1] = 1.0
                if pc < -2:
                    tensor[98 + 4 * i + 2] = 1.0
                    if pc < -3:
                        tensor[98 + 4 * i + 3] = (pc + 3.0) / 2.0
    tensor[96] = board[0] / 2.0
    tensor[97] = a / 15.0
    tensor[194] = board[25] / 2.0
    tensor[195] = b / 15.0
    tensor[196] = 1 if player_1 else 0
    tensor[197] = 0 if player_1 else 1
