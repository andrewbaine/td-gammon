import td_gammon
import torch
import backgammon
import b2
import random

layers = [198, 40, 4]

good = td_gammon.Network(*layers)
good.load_state_dict(torch.load("model.3900.pt"))


def roll():
    return random.randint(1, 6)


mc = b2.MoveComputer()
tensor = torch.as_tensor([0 for _ in range(198)], dtype=torch.float)

with torch.no_grad():
    for i in range(1):
        board = backgammon.make_board()
        scratch_board = [x for x in board]

        i_am_player_1 = int(input("Are you first? 1/0 for Yes/No: "))
        player_1 = False
        while True:
            player_1 = not player_1
            my_turn = player_1 if i_am_player_1 else (not player_1)
            print(backgammon.to_str(board))
            if my_turn:
                d1 = int(input("Enter d1: "))
                d2 = int(input("Enter d2: "))
                min = None
                best_move = None
                moves = mc.compute_moves((board, player_1), (d1, d2))

                for move in moves:
                    for i, x in enumerate(board):
                        scratch_board[i] = x
                    backgammon.unchecked_move(scratch_board, move, player_1=player_1)
                    td_gammon.observe((scratch_board, (not player_1)), tensor)
                    y = torch.dot(
                        good(tensor),
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
                    print("Rolled", (d1, d2), "; played ", best_move)
                    backgammon.unchecked_move(board, best_move, player_1=player_1)
            else:
                line = input(("Enter move: "))
                tokens = [int(x) for x in line.split()]
                move = []
                i = 0
                while i < len(tokens):
                    move.append((tokens[i], tokens[i + 1]))
                    i += 2

                backgammon.unchecked_move(board, move, player_1=player_1)
