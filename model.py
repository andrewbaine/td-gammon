import torch

import torch.nn as nn
import backgammon

import network


class Trainer:
    def __init__(self, model, α=0.10, λ=0.7):
        self.α = α
        self.λ = λ
        self.nn = network.with_utility(model)
        self.eligibility_trace = [
            (w, torch.zeros_like(w, requires_grad=False)) for w in model.parameters()
        ]

    def v(self, observation):
        return self.nn(torch.tensor(observation))

    def train(self, v_next, observation):
        self.nn.zero_grad()
        v = self.nn(torch.tensor(observation))
        v.backward()
        with torch.no_grad():
            δ = v_next - v.item()  # td error
            αδ = self.α * δ  # learning rate * td error
            for w, e in self.eligibility_trace:
                e.mul_(self.λ)
                e.add_(w.grad)
                w.add_(torch.mul(e, αδ))


def best(bck, observer, gamestate, dice, network):
    (_, player_1) = gamestate
    moves = bck.allowed_moves = bck.available_moves(gamestate, dice)
    best = None
    best_move = None
    for move in moves:
        s = bck.next(gamestate, move)
        o = observer.observe(s)
        y = network(torch.tensor(o, dtype=torch.float)).item()
        if best is None or ((y > best) if player_1 else (y < best)):
            best = y
            best_move = move
    return best_move


def td_episode(bck, observer, trainer, num):
    # https://medium.com/clique-org/td-gammon-algorithm-78a600b039bb
    state = bck.s0(player_1=(num % 2 == 0))
    dice = backgammon.first_roll()

    t = 0
    # print("===== begin =====")
    while True:
        (board, player_1) = state
        # print(backgammon.to_str(board))
        # print(t, "player", 1 if player_1 else 2, "to play", dice)
        o = observer.observe(state)
        done = bck.done(state)
        if done:
            # print("done", done)
            trainer.train(done, o)
            return (t, done)
        else:
            with torch.no_grad():
                best_move = best(bck, observer, state, dice, trainer.nn)
                # print("playing", best_move)
            state = bck.next(state, best_move)
            next_observation = observer.observe(state)
            v_next = trainer.v(next_observation).item()
            trainer.train(v_next, o)
            dice = backgammon.roll()
        t += 1
