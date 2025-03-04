import backgammon
import b2

board = backgammon.make_board()

player_1 = True
mc = b2.MoveComputer()

color = backgammon.Color.Dark

(d1, d2) = backgammon.first_roll()
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
        allowed_moves = mc.compute_moves((board, True, (d1, d2)))
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
    (d1, d2) = backgammon.roll()
    color = (
        backgammon.Color.Dark
        if color == backgammon.Color.Light
        else backgammon.Color.Light
    )
