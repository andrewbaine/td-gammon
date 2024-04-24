from itertools import count
import torch
from . import backgammon


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

    def __init__(self, done_checker, agent, eligibility_trace):
        self.starting_position = torch.tensor(
            [backgammon.make_board() + [0]], dtype=torch.float
        )
        self.done_checker = done_checker
        self.agent = agent
        self.eligibility_trace = eligibility_trace

    def s0(self):
        (d1, d2) = first_roll()
        self.starting_position[0][26] = 1 if d1 > d2 else 0
        return (self.starting_position, (d1, d2))

    def episode(self):
        # https://medium.com/clique-org/td-gammon-algorithm-78a600b039bb
        (board, dice) = self.s0()
        for i in count():
            v = self.agent.evaluate(board).squeeze()
            done = self.done_checker.check(board)
            if done:
                self.eligibility_trace.update(v, done)
                return (i, done)
            with torch.no_grad():
                (v_next, board_next) = self.agent.next(board, dice)
            self.eligibility_trace.update(v, v_next.squeeze())
            board = board_next
            v = v_next
            dice = roll()
