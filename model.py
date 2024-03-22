import torch

import random
import torch.nn as nn
import backgammon
import b2
import td_gammon

observe = td_gammon.observe


class Trainer(nn.Sequential):
    def __init__(self, model):
        equity = nn.Linear(4, 1, bias=False)
        equity.weight = nn.Parameter(
            torch.tensor([x for x in td_gammon.utility_tensor], dtype=torch.float32)
        )
        super().__init__(model, equity)

    def reset(self):
        for m in reversed(self):
            m.weight.copy_(
                torch.tensor([x for x in td_gammon.utility_tensor], dtype=torch.float32)
            )
            break


def best(gamestate, scratch_board, tensor, moves, trainer):
    (board, player_1) = gamestate
    best = None
    best_move = None
    for move in moves:
        for j, x in enumerate(board):
            scratch_board[j] = x
        backgammon.unchecked_move(scratch_board, player_1)
        td_gammon.observe((scratch_board, not player_1), tensor)
        y = trainer(tensor).item()
        if best is None or ((y > best) if player_1 else (y < best)):
            best = y
            best_move = move
    return best_move


def roll():
    return random.randint(1, 6)


def episode(trainer):
    α = 0.10
    λ = 0.7

    board = backgammon.make_board()
    scratch_board = [x for x in board]
    mc = b2.MoveComputer()

    tensor = torch.as_tensor([0 for _ in range(198)], dtype=torch.float)

    et = [torch.zeros_like(w, requires_grad=False) for w in trainer.parameters()]

    d1 = roll()
    d2 = roll()
    while d2 == d1:
        d1 = roll()
        d2 = roll()

    t = 0
    player_1 = d1 > d2
    while True:
        observe((board, player_1), tensor)
        trainer.zero_grad()
        v = trainer(tensor)
        v.backward()
        for i, weights in enumerate(trainer.parameters()):  # update e_t+1 based on e_t
            if weights.grad is None:
                raise Exception()
            e = et[i]
            e.mul_(λ)
            e.add_(weights.grad)
        with torch.no_grad():
            ## did the other guy just win?!
            loss = True
            sum = 0
            for x in board:
                if player_1:
                    if x < 0:  # my opponent still has a piece
                        loss = False
                        break
                    elif x > 0:  # count my pips
                        sum += x
                else:
                    if x > 0:
                        loss = False
                        break
                    elif x < 0:
                        sum -= x
            if loss:
                # was I gammoned?
                v_next = (
                    (-1 if sum < 15 else -2) if player_1 else (1 if sum < 15 else 2)
                )
                r = 0
                αδ = α * (v_next - v)
                for i, weights in enumerate(trainer.parameters()):
                    weights.add_(torch.mul(et[i], αδ))
                trainer.reset()
                return (t, v_next)
            else:
                allowed_moves = mc.compute_moves((board, player_1), (d1, d2))
                minimum = None
                best_move = None
                random.shuffle(allowed_moves)
                best_move = best(
                    (board, player_1), scratch_board, tensor, allowed_moves, trainer
                )
                if best_move is not None:
                    backgammon.unchecked_move(board, best_move, player_1=player_1)

                # after playing the best move, compute v_next, the equity to p1 at the next phase
                player_1 = not player_1
                observe((board, player_1), tensor)
                v_next = trainer(tensor).item()
                αδ = α * (v_next - v)
                for i, weights in enumerate(trainer.parameters()):
                    weights.add_(torch.mul(et[i], αδ))
                trainer.reset()
                d1 = roll()
                d2 = roll()
        t += 1


if __name__ == "__main__":
    layers = [198, 40, 4]
    network = td_gammon.Network(*layers)
    trainer = Trainer(network)

    n_episodes = 100000
    results = [0, 0, 0, 0]
    lengths = []
    for i in range(0, n_episodes):
        if i % 100 == 0:
            torch.save(network.state_dict(), "model.{i}.pt".format(i=i))
        (n, result) = episode(trainer)
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
    torch.save(network.state_dict(), "model.final.pt")
