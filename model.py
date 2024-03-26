import torch

import backgammon

import network


class Trainer:
    def __init__(self, bck, model, observe, α=0.10, λ=0.7):
        self._bck = bck
        self.α = α
        self.λ = λ
        self.nn = network.with_utility(model)
        self.eligibility_trace = [
            (w, torch.zeros_like(w, requires_grad=False)) for w in model.parameters()
        ]
        self.observe = observe

    def v(self, state):
        tensor = self.observe(state)
        return self.nn(tensor)

    def train(self, v_next, state):
        self.nn.zero_grad()
        tensor = self.observe(state)
        v = self.nn(tensor)
        v.backward()
        with torch.no_grad():
            δ = v_next - v.item()  # td error
            αδ = self.α * δ  # learning rate * td error
            for w, e in self.eligibility_trace:
                e.mul_(self.λ)
                e.add_(w.grad)
                w.add_(torch.mul(e, αδ))

    def best(self, state, dice):
        (_, player_1) = state
        moves = self._bck.available_moves(state, dice)
        best = None
        best_move = None
        for move in moves:
            s = self._bck.next(state, move)
            tensor = self.observe(s)
            y = self.nn(tensor).item()
            if best is None or ((y > best) if player_1 else (y < best)):
                best = y
                best_move = move
        return best_move

    def td_episode(self, i):
        # https://medium.com/clique-org/td-gammon-algorithm-78a600b039bb
        state = self._bck.s0(player_1=(i % 2 == 0))
        dice = first_roll()

        t = 0
        while True:
            done = self._bck.done(state)
            if done:
                # print("done", done)
                self.train(done, state)
                return (t, done)
            else:
                with torch.no_grad():
                    best_move = self.best(state, dice)
                    # print("playing", best_move)
                next_state = self._bck.next(state, best_move)
                tensor = self.observe(next_state)
                v_next = self.nn(tensor).item()
                self.train(v_next, state)
                dice = roll()
                state = next_state
            t += 1


def roll():
    return tuple(torch.randint(1, 7, (2,)).tolist())


def first_roll():
    while True:
        (d1, d2) = roll()
        if d1 != d2:
            return (d1, d2)
