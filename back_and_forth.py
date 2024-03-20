
import backgammon
import b2

board = backgammon.make_board()
board = [0, 3, 2, 4, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, -3, -2, -5, 0, 0]
board = [0, 5, 2, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -2, -5, -1, 0]
board = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0]

player_1 = True
mc = b2.MoveComputer()
while True:
    print(board)
    print(backgammon.to_str(board))
    print(player_1)
    roll_line = input("Roll: ")
    roll = [int(x) for x in roll_line.split()]
    print(roll)
    
    allowed_moves = mc.compute_moves(board, roll, player_1=player_1)
    for i, x in enumerate(allowed_moves):
        print(i, x)
    if allowed_moves:
        selection = int(input("Move: "))
        move = allowed_moves[selection]
        backgammon.unchecked_move(board, move, player_1=player_1)
    else:
        print("no legal moves")
    player_1 = not player_1

