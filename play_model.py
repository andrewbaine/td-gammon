import td_gammon
import torch
import backgammon
import b2
import random

layers = [198, 40, 4]

good = td_gammon.Network(*layers)
good.load_state_dict(torch.load("model.500.pt"))

bad = td_gammon.Network(*layers)
bad.load_state_dict(torch.load("model.5600.pt"))


def roll():
    return random.randint(1, 6)


mc = b2.MoveComputer()
tensor = torch.as_tensor([0 for _ in range(198)], dtype=torch.float)

results = [0, 0, 0, 0]

with torch.no_grad():
    for i in range(500):
        board = backgammon.make_board()
        scratch_board = [x for x in board]
        d1 = roll()
        d2 = roll()

        while d2 == d1:
            d1 = roll()
            d2 = roll()

        player_1 = d1 > d2

        while True:
            # did I lose?
            loss = True
            sum = 0
            if player_1:
                for i, x in enumerate(board):
                    if x < 0:
                        loss = False
                        break
                    elif x > 0:
                        sum += x
            else:
                for i, x in enumerate(board):
                    if x > 0:
                        loss = False
                        break
                    elif x < 0:
                        sum -= x
            if loss:
                if sum == 15:
                    # gammon
                    results[0 if player_1 else 3] += 1
                else:
                    results[1 if player_1 else 2] += 1
                break
            moves = mc.compute_moves((board, player_1), (d1, d2))
            network = good if player_1 else bad
            min = None
            best_move = None
            for move in moves:
                for i, x in enumerate(board):
                    scratch_board[i] = x
                backgammon.unchecked_move(scratch_board, move, player_1=player_1)
                td_gammon.observe((scratch_board, (not player_1)), tensor)
                y = torch.dot(
                    network(tensor),
                    torch.tensor(
                        [x for x in td_gammon.utility_tensor],
                        requires_grad=False,
                        dtype=torch.float,
                    ),
                ).item()
                if min is None or y < min:
                    min = y
                    best_move = move
            if best_move:
                backgammon.unchecked_move(board, best_move, player_1=player_1)

            player_1 = not player_1
            d1 = roll()
            d2 = roll()
        print(results)
