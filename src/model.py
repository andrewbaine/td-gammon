import torch

import network
import policy


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
        self._policy = policy.Policy_1_ply(bck, observe, self.nn)

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

    def best(self, state):
        return self._policy.choose_action(state)

    def td_episode(self, i):
        # https://medium.com/clique-org/td-gammon-algorithm-78a600b039bb
        state = self._bck.s0()

        t = 0
        while True:
            done = self._bck.done(state)
            if done:
                self.train(done, state)
                return (t, done)
            else:
                with torch.no_grad():
                    best_move = self.best(state)
                next_state = self._bck.next(state, best_move)
                tensor = self.observe(next_state)
                v_next = self.nn(tensor).item()
                self.train(v_next, state)
                state = next_state
            t += 1
