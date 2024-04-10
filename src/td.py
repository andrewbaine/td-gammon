from itertools import count
import torch


def roll():
    d1 = torch.randint(1, 7, (1,), requires_grad=False, device="cpu").item()
    d2 = torch.randint(1, 7, (1,), requires_grad=False, device="cpu").item()
    return (d1, d2)


def first_roll():
    while True:
        (d1, d2) = roll()
        if d1 != d2:
            return (d1, d2)


import eligibility_trace


class TD:

    def __init__(
        self,
        board,
        move_checker,
        agent,
        eligibility_trace: eligibility_trace.ElibilityTrace,
    ):
        self.board = board
        self.move_checker = move_checker
        self.agent = agent
        self.eligibility_trace = eligibility_trace

    def s0(self):
        (d1, d2) = first_roll()
        player_1 = d1 > d2
        return (self.board, player_1, (d1, d2))

    def episode(self):
        # https://medium.com/clique-org/td-gammon-algorithm-78a600b039bb
        state = self.s0()
        for i in count():
            (board, player_1, _) = state
            print("board.device in episode", board.state)
            v = self.agent.evaluate(state)
            done = self.move_checker.check(board)
            if done:
                self.eligibility_trace.update(v, done)
                return (i, done)
            with torch.no_grad():
                (v_next, board_next) = self.agent.next(state)
            self.eligibility_trace.update(v, v_next)
            state = (board_next, not player_1, roll())
