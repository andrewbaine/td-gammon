from itertools import count
import torch
import network


def roll():
    d1 = torch.randint(1, 7, (1,), requires_grad=False, device="cpu").item()
    d2 = torch.randint(1, 7, (1,), requires_grad=False, device="cpu").item()
    return (d1, d2)


def first_roll():
    while True:
        (d1, d2) = roll()
        if d1 != d2:
            return (d1, d2)


class TD:

    def __init__(
        self,
        board,
        move_checker,
        agent,
        nn,
        α=0.10,
        λ=0.7,
        device=torch.device("cuda"),
    ):
        self.board = board
        self.move_checker = move_checker
        self.agent = agent
        self.α = α
        self.λ = λ
        self.eval = eval
        self.eligibility_trace = [
            (w, torch.zeros_like(w, requires_grad=False, device=device))
            for w in nn.parameters()
        ]

    def train(self, v_next, state):
        v = self.agent.evaluate(state)
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
            done = self.move_checker.check(board)
            if done:
                self.train(done, state)
                return (i, done)
            with torch.no_grad():
                (v_next, board_next) = self.agent.next(state)
            self.train(v_next, state)
            state = (board_next, not player_1, roll())
