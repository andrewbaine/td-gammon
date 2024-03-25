import backgammon
import b2
import torch
import random


class Backgammon:
    def __init__(self):
        self.mc = b2.MoveComputer()
        self.board = backgammon.make_board()
        self.player_1 = True

    def s0(self, player_1=None):
        if player_1 is None:
            player_1 = random.random() < 0.5
        return (tuple(self.board), player_1)

    def available_moves(self, state, roll):
        return self.mc.compute_moves(state, roll)

    def next(self, state, action):
        (board, player_1) = state
        if action:
            scratch = [x for x in board]
            backgammon.unchecked_move(scratch, action, player_1=player_1)
            return (tuple(scratch), not player_1)
        else:
            return (board, not player_1)

    def done(self, state):
        (board, player_1) = state
        my_checker_count = 0
        if player_1:
            for x in board:
                if x < 0:  # i didnt lose
                    return None
                my_checker_count += x
            return -1 if my_checker_count < 15 else -2
        else:
            for x in board:
                if x > 0:  # i didnt lose
                    return None
                my_checker_count -= x
            return 1 if my_checker_count < 15 else 2
