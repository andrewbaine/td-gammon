import backgammon
import b2
import torch


class Backgammon:
    def __init__(self):
        self.mc = b2.MoveComputer()
        self.board = backgammon.make_board()
        self.player_1 = True

    def s0(self):
        return (tuple(self.board), True)

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


class Teasoro198:
    def __init__(self):
        super().__init__()

    def observe(self, state):
        (board, player_1) = state
        tensor = torch.as_tensor([0 for _ in range(198)], dtype=torch.float)
        a = 15.0
        b = 15.0
        for i, pc in enumerate(board):
            if i == 0:
                if pc < 0:
                    tensor[192] = pc / -2.0
                elif pc > 0:
                    raise Exception("unexpected at i == 0")
            elif i < 25:
                t = 8 * (i - 1)
                for j in range(0, 8):
                    tensor[t + j] = 0.0
                if pc > 0:
                    a -= pc
                    tensor[t] = 1.0
                    if pc > 1:
                        tensor[t + 1] = 1.0
                        if pc > 2:
                            tensor[t + 2] = 1.0
                            if pc > 3:
                                tensor[t + 3] = (pc - 3.0) / 2.0
                elif pc < 0:
                    b += pc
                    tensor[t + 4] = 1.0
                    if pc < -1:
                        tensor[t + 5] = 1.0
                        if pc < -2:
                            tensor[t + 6] = 1.0
                            if pc < -3:
                                tensor[t + 7] = (-1 * pc - 3.0) / 2.0
                            pass
            elif i == 25:
                if pc < 0:
                    raise Exception("unexpected at i == 25")
                elif pc > 0:
                    tensor[193] = pc / 2.0
            else:
                raise Exception("impossible")

        tensor[194] = a / 15.0
        tensor[195] = b / 15.0
        tensor[196] = 1 if player_1 else 0
        tensor[197] = 0 if player_1 else 1
        return tensor
