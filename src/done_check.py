import torch


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

    def is_flag(self, x):
        assert self.n0.equal(x) or self.n1.equal(x)

    def check(self, board):
        a = torch.maximum(board, self.zero)
        a_count = torch.matmul(a, self.one_column)
        sign_a_count = torch.minimum(a_count, self.n1)
        a_win_point = self.n1.subtract(sign_a_count)

        b = torch.minimum(board, self.zero)
        b_count = torch.matmul(b, self.one_column)
        b_any_borne_off = torch.minimum(b_count + self.fifteen, self.n1)  # 0 or 1
        self.is_flag(b_any_borne_off)
        b_gammoned = self.n1 - b_any_borne_off  # 0 or 1
        self.is_flag(b_gammoned)
        a_gammon_point = a_win_point * b_gammoned  # 0 or 1
        self.is_flag(b_gammoned)

        b_backgammoned = torch.minimum(
            torch.matmul(b, self.b_backgammoned), self.n1
        )  # 0 or 1
        self.is_flag(b_backgammoned)

        a_backgammon_point = a_gammon_point * b_backgammoned
        self.is_flag(a_backgammon_point)

        assert self.n0.equal(a_gammon_point) or self.n1.equal(a_win_point)
        assert self.n0.equal(a_backgammon_point) or self.n1.equal(a_gammon_point)

        sign_b_count = torch.maximum(b_count, self.n1_neg)
        assert self.n1_neg.equal(sign_b_count) or self.n0.equal(sign_b_count)
        b_win_point = self.n1_neg.subtract(sign_b_count)
        assert self.n1_neg.equal(b_win_point) or self.n0.equal(b_win_point)

        a_borne_off_count = a_count.sub(self.fifteen).neg()
        a_any_borne_off = torch.minimum(a_borne_off_count, self.n1)
        self.is_flag(a_any_borne_off)

        a_gammoned = self.n1 - a_any_borne_off
        self.is_flag(a_gammoned)

        b_gammon_point = b_win_point * a_gammoned
        assert self.n1_neg.equal(b_gammon_point) or self.n0.equal(b_gammon_point)

        a_backgammoned = torch.minimum(torch.matmul(a, self.a_backgammoned), self.n1)
        self.is_flag(a_backgammoned)

        b_backgammon_point = b_gammon_point * a_backgammoned

        assert self.n0.equal(b_backgammon_point) or self.n1_neg.equal(b_gammon_point)
        assert self.n0.equal(b_gammon_point) or self.n1_neg.equal(b_win_point)
        assert b_backgammon_point == 0 or b_gammon_point == -1

        assert a_win_point == 0 or b_win_point == 0
        return (
            a_win_point
            + a_gammon_point
            + a_backgammon_point
            + b_win_point
            + b_gammon_point
            + b_backgammon_point
        )
