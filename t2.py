import torch


def make_tesauro(device):
    v = []
    v2 = []
    v3 = []
    w_1 = []
    for i in range(26):
        v2.append([0 if z != 194 else 1 for z in range(198)])
        v3.append([0 if z != 195 else 1 for z in range(198)])
        row = []
        v.append(row)
        for j in range(1, 25):
            for k in range(0, 4):
                row.append(1 if i == j else 0)
                row.append(-1 if i == j else 0)
                if i == 0:
                    w_1.append(-k)
                    w_1.append(-k)

        row.append(-1 if i == 0 else 0)  # player 2 bar
        row.append(1 if i == 25 else 0)  # player 1 bar
        row.append(1)  # count white pieces
        row.append(1)  # count black pieces
        row.append(0)  # this will be player 1's turn
        row.append(0)  # this will be player 2's turn
        if i == 0:
            for _ in range(4):
                w_1.append(0)
            w_1.append(1)
            w_1.append(0)
        assert len(row) == 198
        assert len(w_1) == 198
    w_2 = [x if i < 196 else (1 - x) for i, x in enumerate(w_1)]
    return (
        torch.tensor(v, dtype=torch.float, device=device),
        torch.tensor(v2, dtype=torch.float, device=device),
        torch.tensor(v3, dtype=torch.float, device=device),
        torch.tensor(w_1, dtype=torch.float, device=device),
        torch.tensor(w_2, dtype=torch.float, device=device),
    )


class Encoder:
    def __init__(self, device):
        v, v2, v3, w_1, w_2 = make_tesauro(device=device)
        self.v = v
        self.v2 = v2
        self.v3 = v3
        self.zero = torch.tensor(
            [0 for _ in range(198)], dtype=torch.float, device=device
        )
        self.zero_board = torch.tensor(
            [0 for _ in range(26)], dtype=torch.float, device=device
        )

        self.w_1 = w_1
        self.w_2 = w_2

    def encode(self, board, player_1):
        x = torch.matmul(board, self.v)
        x = torch.maximum(x, self.zero)

        x = torch.add(x, self.w_1 if player_1 else self.w_2)
        x = torch.maximum(x, self.zero)

        x = torch.add(x, torch.matmul(torch.maximum(board, self.zero_board), self.v2))
        x = torch.add(x, torch.matmul(torch.minimum(board, self.zero_board), self.v3))

        return x


import backgammon

enc = Encoder(device="cpu")
board = torch.tensor(backgammon.make_board(), dtype=torch.float)
x = enc.encode(board, False)
print(x.tolist())
