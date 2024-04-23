import torch

import backgammon_env


class Agent:
    pass


class RandomAgent(Agent):
    def __init__(self):
        self.bck = backgammon_env.Backgammon()

    def decide_action(self, state):
        ms = self.bck.available_moves(state)
        if not ms:
            return None
        i = torch.randint(0, len(ms) - 1, (1,))[0]
        return ms[i]


class OnePlyAgent(Agent):
    def __init__(self, nn, move_tensors):
        self.nn = nn
        self.move_tensors = move_tensors

    def evaluate(self, board):
        (_, n) = board.size()
        assert n == 27
        return self.nn(board)

    def next(self, board, dice):
        move_vectors = self.move_tensors.compute_move_vectors(board, dice)
        next_states = torch.add(move_vectors, board)
        (_, n) = next_states.size()
        assert n == 27
        utilities = self.evaluate(next_states)
        us = (2 * (board[:, [26]].unsqueeze(1)) - 1) * utilities
        index = torch.argmax(us)
        utility_next = utilities[index]
        board_next = next_states[index]
        return (utility_next, board_next)

    def decide_action(self, state, dice):
        state_old = state
        (_, board_next) = self.next(state, dice)

        bbb = [int(x) for x in board_next.tolist()[:26]]
        bck = backgammon_env.Backgammon()
        for m in bck.available_moves(state_old):
            (board, _, _) = bck.next((state_old[0:26], state_old[26], dice), m)
            if board == bbb:
                return m
        assert False
