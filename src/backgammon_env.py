import random

import b2
import backgammon


class Backgammon:
    def __init__(self, roll=lambda: random.randint(1, 6)):
        self.mc = b2.MoveComputer()
        self.board = backgammon.make_board()
        self.roll = roll

    def s0(self):
        dice = None
        while True:
            dice = (self.roll(), self.roll())
            (d1, d2) = dice
            if d1 != d2:
                break
        player_1 = d1 > d2
        return (tuple(self.board), player_1, dice)

    def available_moves(self, state):
        return self.mc.compute_moves(state)

    def next(self, state, action):
        (board, player_1, dice) = state
        dice = (self.roll(), self.roll())
        if action:
            scratch = [x for x in board]
            backgammon.unchecked_move(scratch, action, player_1=player_1)
            return (tuple(scratch), not player_1, dice)
        else:
            return (board, not player_1, dice)

    def done(self, state):
        (board, player_1, dice) = state
        my_checker_count = 0
        backgammoned = 0
        if player_1:
            for i, x in enumerate(board):
                if x < 0:  # i didnt lose
                    return 0
                if i > 18:
                    backgammoned += x
                my_checker_count += x
            return -1 if my_checker_count < 15 else (-3 if backgammoned else -2)
        else:
            for i, x in enumerate(board):
                if x > 0:  # i didnt lose
                    return 0
                if i < 7:
                    backgammoned -= x
                my_checker_count -= x
            return 1 if my_checker_count < 15 else (3 if backgammoned else 2)
