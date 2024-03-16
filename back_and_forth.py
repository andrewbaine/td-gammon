
import backgammon
import sneaky

board = backgammon.make_board()
board = [0, 3, 2, 4, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, -3, -2, -5, 0, 0]
board = [0, 5, 2, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -2, -5, -1, 0]
board = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -2, 0]

player_1 = True
while True:
    print(board)
    print(backgammon.to_str(board))
    roll_line = input("Roll: ")
    roll = [int(x) for x in roll_line.split()]
    print(roll)
    
    allowed_moves = sneaky.moves(sneaky.from_board(board, player_1=player_1), roll[0], roll[1], player_1=player_1)
    for i, x in enumerate(allowed_moves):
        print(i, [(x.source, x.destination) for x in x])
    if allowed_moves:
        selection = int(input("Move: "))
        move = allowed_moves[selection]
        backgammon.unchecked_move(board, move, player_1=player_1)
    else:
        print("no legal moves")
    player_1 = not player_1
