import backgammon
import sneaky

def test_6_5():
    board = backgammon.make_board()
    s = sneaky.from_board(board)
    expected = sneaky.canonically_sorted_moves([sneaky.Move(a, b) for (a, b) in moves] for moves in [
        [(24, 18), (18, 13)],
        [(24, 18), (13, 8)],
        [(24, 18), (8, 3)],
        [(13, 7), (13, 8)],
        [(13, 7), (8, 3)],
        [(13, 7), (7, 2)],
        [(8, 2), (8, 3)],
        [(13, 8), (8, 2)],
    ])
    for (d1, d2) in [(6, 5), (5, 6)]:
        assert sneaky.moves(s, d1, d2) == expected
