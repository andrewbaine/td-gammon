import backgammon
from itertools import count
import torch
import network


def roll():
    d1 = torch.randint(1, 7, (1,)).item()
    d2 = torch.randint(1, 7, (1,)).item()
    return (d1, d2)


def first_roll():
    while True:
        (d1, d2) = roll()
        if d1 != d2:
            return (d1, d2)


class TD:

    def __init__(self, board, move_checker, move_tensors, nn, encoder, α=0.10, λ=0.7):
        self.board = board
        self.move_checker = move_checker
        self.move_tensors = move_tensors
        self.encoder = encoder
        self.α = α
        self.λ = λ
        self.eval = eval
        self.eligibility_trace = [
            (w, torch.zeros_like(w, requires_grad=False)) for w in nn.parameters()
        ]
        self.nn = network.with_utility(nn)

    def train(self, v_next, state):
        (board, player_1, _) = state
        te = self.encoder.encode(board, player_1)
        v = self.nn(te)
        v.backward()
        with torch.no_grad():
            δ = v_next - v  # td error
            αδ = (self.α * δ).squeeze()  # learning rate * td error
            for w, e in self.eligibility_trace:
                e.mul_(self.λ)
                e.add_(w.grad)
                w.add_(torch.mul(e, αδ))

    def s0(self):
        (d1, d2) = first_roll()
        player_1 = d1 > d2
        return (self.board, player_1, (d1, d2))

    def episode(self):
        # https://medium.com/clique-org/td-gammon-algorithm-78a600b039bb
        state = self.s0()
        for i in count():
            (board, player_1, _) = state
            print(backgammon.to_str(board.tolist()))
            done = self.move_checker.check(board)
            if done:
                self.train(done, state)
                print(done)
                return (i, done)
            with torch.no_grad():
                move_vectors = self.move_tensors.compute_move_vectors(state)
                next_states = torch.add(move_vectors, board)
                tesauro_encoded = self.encoder.encode(next_states, not player_1)
                vs = self.nn(tesauro_encoded)
                index = (torch.argmax if player_1 else torch.argmin)(vs)
                v_next = vs[index]
                board_next = next_states[index]
            self.train(v_next, state)
            state = (board_next, not player_1, roll())
