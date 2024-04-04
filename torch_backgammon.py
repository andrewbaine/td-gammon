import backgammon
import torch


class Backgammon:
    def __init__(self, board=None, dice=None, player_1=None, seed=None):
        if seed:
            torch.manual_seed(seed)

        if board:
            self.board = torch.tensor(board, dtype=torch.int8)
        else:
            self.board = torch.tensor(backgammon.make_board(), dtype=torch.int8)

        if dice:
            self.dice = dice
        else:
            while True:
                dice = torch.randint(1, 7, (2,))
                d1 = dice[0].item()
                d2 = dice[1].item()
                if d1 != d2:
                    self.dice = (d1, d2)
                    break

        if player_1 is not None:
            self.player_1 = player_1
        else:
            (d1, d2) = self.dice
            assert d1 != d2
            self.player_1 = d1 < d2

    def next(self, move):
        
