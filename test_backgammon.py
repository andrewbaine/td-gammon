import backgammon

def test_6_5():
    board = backgammon.make_board()
    expected = [
        [(24, 18), (18, 13)],
        [(24, 18), (13, 8)],
        [(24, 18), (8, 3)],
        [(13, 7), (13, 8)],
        [(13, 7), (8, 3)],
        [(13, 7), (7, 2)],
        [(8, 2), (8, 3)],
        [(13, 8), (8, 2)],
    ]
    for roll in [(6, 5), (5, 6)]:
        assert backgammon.allowed_moves(board, roll) == expected


