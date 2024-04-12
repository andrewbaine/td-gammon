import torch

from torch import maximum, matmul, minimum


class Donecheck:
    def __init__(self):
        self.zero = torch.tensor([0 for _ in range(26)], dtype=torch.float)
        self.one_column = torch.tensor([[1] for _ in range(26)], dtype=torch.float)

        self.a_backgammoned = torch.tensor(
            [[1 if i > 18 else 0] for i in range(26)], dtype=torch.float
        )
        self.b_backgammoned = torch.tensor(
            [[-1 if i < 7 else 0] for i in range(26)], dtype=torch.float
        )
        self.fifteen = torch.tensor([[15]], dtype=torch.float)
        self.fifteen_negated = torch.tensor([[-15]], dtype=torch.float)
        self.n0 = torch.tensor([[0]], dtype=torch.float)
        self.n1 = torch.tensor([[1]], dtype=torch.float)
        self.n1_neg = self.n1.neg()

    def to_(self, device):
        self.zero = self.zero.to(device=device)
        self.one_column = self.one_column.to(device=device)
        self.a_backgammoned = self.a_backgammoned.to(device=device)
        self.b_backgammoned = self.b_backgammoned.to(device=device)
        self.fifteen = self.fifteen.to(device=device)
        self.fifteen_negated = self.fifteen_negated.to(device=device)
        self.n0 = self.n0.to(device=device)
        self.n1 = self.n1.to(device=device)
        self.n1_neg = self.n1_neg.to(device=device)

    def check(self, board):
        a = maximum(board, self.zero)
        a_count = matmul(a, self.one_column)
        sign_a_count = minimum(a_count, self.n1)
        a_win_point = self.n1.subtract(sign_a_count)

        b = minimum(board, self.zero)
        b_count = matmul(b, self.one_column)
        b_any_borne_off = minimum(b_count + self.fifteen, self.n1)  # 0 or 1

        b_gammoned = self.n1 - b_any_borne_off  # 0 or 1

        a_gammon_point = a_win_point * b_gammoned  # 0 or 1

        b_backgammoned = minimum(matmul(b, self.b_backgammoned), self.n1)  # 0 or 1

        a_backgammon_point = a_gammon_point * b_backgammoned

        sign_b_count = maximum(b_count, self.n1_neg)
        b_win_point = self.n1_neg.subtract(sign_b_count)

        a_borne_off_count = a_count.sub(self.fifteen).neg()
        a_any_borne_off = minimum(a_borne_off_count, self.n1)

        a_gammoned = self.n1 - a_any_borne_off

        b_gammon_point = b_win_point * a_gammoned

        a_backgammoned = minimum(matmul(a, self.a_backgammoned), self.n1)

        b_backgammon_point = b_gammon_point * a_backgammoned

        return (
            a_win_point
            + a_gammon_point
            + a_backgammon_point
            + b_win_point
            + b_gammon_point
            + b_backgammon_point
        ).squeeze()
