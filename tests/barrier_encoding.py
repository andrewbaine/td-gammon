import torch


def barrier_matrix(b):
    ms = []

    scales = [
        torch.tensor([(n if b else -n) for _ in range(24)]).diag().tolist()
        for n in range(1, 8)
    ]

    for n in range(1, 8):
        row = []
        ms.append(row)
        for i in range(24):
            x = []
            row.append(x)
            for j in range(24):
                x.append(1 if (-1 < ((i - j) if b else (j - i)) < n) else 0)
    return (
        torch.tensor(ms, dtype=torch.float),
        torch.tensor(scales, dtype=torch.float),
    )


class Barrier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.barrier_additions = torch.tensor(
            [[n - 1 for _ in range(24)] for n in range(1, 8)], dtype=torch.float
        )
        (self.player_1_barrier_matrix, self.barrier_scales_player_1) = barrier_matrix(
            True
        )
        (self.player_2_barrier_matrix, self.barrier_scales_player_2) = barrier_matrix(
            False
        )
        self.zero24 = torch.tensor([0 for _ in range(24)], dtype=torch.float)
        self.one24 = torch.tensor([1 for _ in range(24)], dtype=torch.float)

    def forward(self, board):
        #        if board.shape == (26,):
        #            board = board.unsqueeze(0)
        (n, _) = board.shape

        additions = self.barrier_additions
        additions = additions.unsqueeze(1).expand(-1, n, -1)

        points_made = torch.where(board > 1, self.one24, self.zero24)

        m = self.player_1_barrier_matrix
        a = torch.matmul(points_made, m)
        b = torch.sub(a, additions)
        zero24 = self.zero24.unsqueeze(0).expand(n, -1).unsqueeze(0).expand(7, -1, -1)
        c = torch.maximum(b, zero24)
        d = torch.matmul(c, self.barrier_scales_player_1)
        q1 = torch.max(d, dim=0).values

        points_made = torch.where(board < -1, self.one24, self.zero24)
        m = self.player_2_barrier_matrix
        a = torch.matmul(points_made, m)
        b = torch.sub(a, additions)
        c = torch.maximum(b, zero24)
        d = torch.matmul(c, self.barrier_scales_player_2)
        q2 = torch.min(d, dim=0).values

        return q1 + q2


class GreatestBarrier(Barrier):
    def __init__(self):
        super().__init__()
        self.zeros_to_the_right = torch.tensor(
            [
                [1 if i == j else -1 if j == i - 1 else 0 for j in range(24)]
                for i in range(24)
            ],
            dtype=torch.float,
        )
        self.zeros_to_the_left = torch.tensor(
            [
                [1 if i == j else -1 if j == i + 1 else 0 for j in range(24)]
                for i in range(24)
            ],
            dtype=torch.float,
        )

    def forward(self, x):
        y = torch.maximum(x, self.zero24)
        r = torch.matmul(y, self.zeros_to_the_left)
        s = torch.where(r >= y, y, self.zero24)

        y2 = torch.minimum(x, self.zero24)
        r2 = torch.matmul(y2, self.zeros_to_the_right)
        s2 = torch.where(r2 <= y2, y2, self.zero24)
        return s + s2
