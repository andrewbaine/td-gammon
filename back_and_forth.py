import backgammon
import b2

import random


def roll():
    return random.randint(1, 6)


board = backgammon.make_board()

player_1 = True
mc = b2.MoveComputer()

d1 = roll()
d2 = roll()
while d2 == d1:
    d2 = roll()

color = backgammon.Color.Dark
while True:
    print(backgammon.to_str(board, player_1_color=color))
    ## did the other guy just win?!
    loss = True
    sum = 0
    for x in board:
        if x < 0:
            loss = False
            break
        elif x > 0:
            sum += x
    if loss:
        result = [0, 0, 1, 0] if sum < 15 else [0, 0, 0, 1]
        print(result)
        break
    else:
        allowed_moves = mc.compute_moves(board, (d1, d2), player_1=True)
        print("roll:", d1, d2)
        for i, x in enumerate(allowed_moves):
            print(i, x)
        if allowed_moves:
            selection = 0
            selection = int(input("Move: "))
            move = allowed_moves[selection]
            backgammon.unchecked_move(board, move, player_1=True)
        else:
            print("no legal moves")
    backgammon.invert(board)
    d1 = roll()
    d2 = roll()
    color = (
        backgammon.Color.Dark
        if color == backgammon.Color.Light
        else backgammon.Color.Light
    )
