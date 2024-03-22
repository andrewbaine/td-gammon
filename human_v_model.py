import td_gammon
import torch
import backgammon
import b2
import random

layers = [198, 40, 4]

good = td_gammon.Network(*layers)
good.load_state_dict(torch.load("model.700.pt"))


def roll():
    return random.randint(1, 6)


mc = b2.MoveComputer()
tensor = torch.as_tensor([0 for _ in range(198)], dtype=torch.float)

with torch.no_grad():
    for i in range(1):
        board = backgammon.make_board()
        scratch_board = [x for x in board]
        d1 = roll()
        d2 = roll()

        while d2 == d1:
            d1 = roll()
            d2 = roll()

        player_1 = d1 > d2
        human_first = d1 < d2

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
                    print(("human" if player_1 else "robot") + " won by 2 points")

                else:
                    print(("human" if player_1 else "robot") + " won 1 point")
                break
            moves = mc.compute_moves((board, player_1), (d1, d2))
            if player_1:
                min = None
                best_move = None
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
                if human_first:
                    print(backgammon.to_str(board))
                else:
                    backgammon.invert(board)
                    print(backgammon.to_str(board))
                    backgammon.invert(board)
                print("Roll:", (d1, d2))
                if moves:
                    for i, move in enumerate(moves):
                        print(i, move)
                    i = int(input("Select move: "))
                    move = moves[i]
                    backgammon.unchecked_move(board, move, player_1=player_1)
            player_1 = not player_1
            d1 = roll()
            d2 = roll()
