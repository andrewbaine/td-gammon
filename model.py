import torch

import random
import torch.nn as nn
import backgammon
import b2
import td_gammon


class Trainer(nn.Sequential):
    def __init__(self, model):
        utility = nn.Linear(4, 1, bias=False)
        utility.weight = nn.Parameter(
            torch.tensor([x for x in td_gammon.utility_tensor], dtype=torch.float32)
        )
        super().__init__(model, utility)
        self.__reset_utility = lambda _: (
            utility.weight.copy_(
                torch.tensor([x for x in td_gammon.utility_tensor], dtype=torch.float32)
            )
        )

    def reset_utility_tensor(self):
        self.__reset_utility(None)
        # for m in reversed(self):
        #     m.weight.copy_(
        #         torch.tensor([x for x in td_gammon.utility_tensor], dtype=torch.float32)
        #     )
        #     break


def best(bck, observer, gamestate, dice, trainer):
    (_, player_1) = gamestate
    moves = bck.allowed_moves = bck.available_moves(gamestate, dice)
    best = None
    best_move = None
    for move in moves:
        s = bck.next(gamestate, move)
        tensor = observer.observe(s)
        y = trainer(tensor).item()
        if best is None or ((y > best) if player_1 else (y < best)):
            best = y
            best_move = move
    return best_move


def episode(bck, observer, trainer):
    α = 0.10
    λ = 0.7

    state = bck.s0()

    et = [torch.zeros_like(w, requires_grad=False) for w in trainer.parameters()]

    dice = backgammon.first_roll()

    t = 0
    while True:
        tensor = observer.observe(state)
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
            done = bck.done(state)
            if done:
                # was I gammoned?
                v_next = done
                αδ = α * (v_next - v)
                for i, weights in enumerate(trainer.parameters()):
                    weights.add_(torch.mul(et[i], αδ))
                trainer.reset_utility_tensor()
                return (t, v_next)
            else:
                best_move = best(bck, observer, state, dice, trainer)
                state = bck.next(state, best_move)

                tensor = observer.observe(state)
                v_next = trainer(tensor).item()
                αδ = α * (v_next - v)
                for i, weights in enumerate(trainer.parameters()):
                    weights.add_(torch.mul(et[i], αδ))
                trainer.reset_utility_tensor()
                dice = backgammon.roll()
        t += 1


import backgammon_env

if __name__ == "__main__":
    layers = [198, 40, 4]
    network = td_gammon.Network(*layers)
    trainer = Trainer(network)
    observer = backgammon_env.Teasoro198()

    n_episodes = 100000
    results = [0, 0, 0, 0]
    lengths = []
    bck = backgammon_env.Backgammon()
    for i in range(0, n_episodes):
        if (
            (i < 1000 and (i % 100 == 0))
            or (i < 10000 and (i % 500) == 0)
            or (i % 1000 == 0)
        ):
            torch.save(network.state_dict(), "model.{i}.pt".format(i=i))
        (n, result) = episode(bck, observer, trainer)
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
        print(results, n)
    print(lengths, results)
    torch.save(network.state_dict(), "model.final.pt")
