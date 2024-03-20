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

while True:
    print(backgammon.to_str(board))
    allowed_moves = mc.compute_moves(board, (d1, d2), player_1=player_1)
    print("roll:", d1, d2)
    for i, x in enumerate(allowed_moves):
        print(i, x)
    if allowed_moves:
        selection = int(input("Move: "))
        move = allowed_moves[selection]
        backgammon.unchecked_move(board, move, player_1=player_1)
    else:
        print("no legal moves")
    player_1 = not player_1
    d1 = roll()
    d2 = roll()
