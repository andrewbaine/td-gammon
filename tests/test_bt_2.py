import pytest
from torch import tensor

import backgammon
import test_b2
import read_move_tensors
import contextlib
import torch


@pytest.fixture
def move_tensors():
    with (
        torch.cuda.device("cuda")
        if torch.cuda.is_available()
        else contextlib.nullcontext()
    ):
        yield read_move_tensors.MoveTensors(
            "var/move_tensors/current",
        )


@pytest.mark.parametrize("t", test_b2.test_cases)
def tests(move_tensors, t):
    board = backgammon.from_str(t.board, player_1_color=t.player_1_color)
    player_1 = t.player == t.player_1_color
    state = (tensor(board), player_1, t.roll)
    (moves, _) = move_tensors.compute_moves(state)
    assert moves is not None
    moves = moves.tolist()
    moves = [[(a, b) for (a, b) in zip(x[::3], x[1::3])] for x in moves]
    for x in moves:
        x.sort()
        x.reverse()
    moves = [tuple(x) for x in moves]
    moves.sort()
    moves.reverse()
    assert moves == t.expected_moves, t.comment
