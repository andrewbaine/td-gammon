import torch

import random
import torch.nn as nn
import itertools


class Model(nn.Sequential):
    # http://incompleteideas.net/book/first/ebook/node87.html
    def __init__(self, in_features, hidden_features, n_hidden_layers):
        super().__init__(
            *(
                [nn.Linear(in_features, hidden_features), nn.Sigmoid()]
                + list(
                    itertools.chain(
                        *[
                            [nn.Linear(hidden_features, hidden_features), nn.Sigmoid()]
                            for _ in range(n_hidden_layers)
                        ]
                    )
                )
                + [nn.Linear(hidden_features, 4)]
            ),
        )


class Trainer(nn.Sequential):
    def __init__(self, model):
        equity = nn.Linear(4, 1, bias=False)
        weights = torch.tensor([[2, 1, -1, -2]], dtype=torch.float32)

        equity.weight = nn.Parameter(weights)

        super().__init__(model, equity)


import backgammon
import b2

import random


def roll():
    return random.randint(1, 6)


def set_tensor(board, tensor):
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


def episode(model, trainer):

    board = backgammon.make_board()
    scratch_board = [x for x in board]
    mc = b2.MoveComputer()

    tensor = torch.as_tensor([0 for _ in range(196)], dtype=torch.float)

    et = [
        (torch.zeros_like(layer.weight), layer.weight)
        for layer in model
        if isinstance(layer, nn.Linear)
    ]

    d1 = roll()
    d2 = roll()
    while d2 == d1:
        d2 = roll()

    t = 0
    while True:
        set_tensor(board, tensor)
        trainer.zero_grad()
        v = trainer(tensor)
        v.backward()
        for e, weight in et:  # update e_t+1 based on e_t
            if weight.grad is None:
                raise Exception()
            e.mul_(γ)
            e.mul_(λ)
            e.add_(weight.grad)

        ## did the other guy just win?!
        loss = True
        sum = 0
        for x in board:
            if x < 0:
                loss = False
                break
            elif x > 0:
                sum += x
        if loss:
            v_next = -1 if sum < 15 else -2
            r = 0
            αδ = r + α * (γ * v_next - v)
            for e, weight in et:
                if weight.grad is None:
                    raise Exception()
                weight.add_(torch.mul(e, αδ))
            return (t, v_next * (1 if i % 2 == 0 else -1))
        else:

            set_tensor(board, tensor)

            allowed_moves = mc.compute_moves(board, (d1, d2), player_1=True)
            minimum = None
            best_move = None
            random.shuffle(allowed_moves)
            for move in allowed_moves:
                for j, pc in enumerate(board):
                    scratch_board[j] = pc
                backgammon.unchecked_move(scratch_board, move, player_1=True)
                backgammon.invert(scratch_board)
                set_tensor(scratch_board, tensor)
                y = trainer(tensor).item()
                if minimum is None or y < minimum:
                    minimum = y
                    best_move = move
            if best_move is not None:
                backgammon.unchecked_move(board, best_move, player_1=True)

            backgammon.invert(board)
            set_tensor(board, tensor)
            v_next = -1 * trainer(tensor).item()
            r = 0
            αδ = r + α * (γ * v_next - v)
            for e, layer in et:
                with torch.no_grad():
                    layer.weight.add_(torch.mul(e, αδ))
            d1 = roll()
            d2 = roll()
        t += 1


if __name__ == "__main__":
    α = 0.05
    γ = 1.0
    λ = 0.7
    network = Model(196, 80, 2)
    trainer = Trainer(network)

    n_episodes = 100000
    results = [0, 0, 0, 0]
    lengths = []
    for i in range(0, n_episodes):
        if i % 500 == 0:
            torch.save(network.state_dict(), "entire_model.{i}.pt".format(i=i))
        (n, result) = episode(network, trainer)
        lengths.append(n)
        match result:
            case -2:
                results[0] += 1
            case -1:
                results[1] += 1
            case 1:
                results[2] += 1
            case 2:
                results[3] += 1
            case _:
                raise Exception("unexpected")
        print(n)
        print(results)
    print(lengths, results)
    torch.save(network.state_dict(), "entire_model.final.pt")
