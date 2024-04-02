import pytest
from torch import tensor

import backgammon
import bt_2
import test_b2


@pytest.mark.parametrize("t", test_b2.test_cases)
def tests(t):
    board = backgammon.from_str(t.board, player_1_color=t.player_1_color)
    player_1 = t.player == t.player_1_color
    if not player_1:
        backgammon.invert(board)
    (d1, d2) = t.roll
    if d1 != d2:
        moves = bt_2.compute_moves(tensor(board), t.roll)
        print(moves)
        moves = moves.tolist()
        moves = [[(a, b) for (a, b) in zip(x[::3], x[1::3])] for x in moves]
        for x in moves:
            x.sort()
            x.reverse()
        moves = [tuple(x) for x in moves]
        moves.sort()
        moves.reverse()
        assert moves == t.expected_moves, t.comment
    else:
        assert True
