import contextlib

import pytest
from torch import tensor
import torch

import backgammon
import backgammon_env
import read_move_tensors
import test_b2


@pytest.fixture
def move_tensors():
    with (
        torch.cuda.device("cuda")
        if torch.cuda.is_available()
        else contextlib.nullcontext()
    ):
        yield read_move_tensors.MoveTensors()


@pytest.fixture
def bck():
    return backgammon_env.Backgammon()


@pytest.mark.parametrize("t", test_b2.test_cases)
def tests(move_tensors, bck, t):
    board = backgammon.from_str(t.board, player_1_color=t.player_1_color)
    player_1 = t.player == t.player_1_color
    tensor_board = tensor(board)
    state = (tensor_board, player_1, t.roll)
    (vectors) = move_tensors.compute_move_vectors(state)
    s = set()
    for v in vectors:
        b = tuple(int(x) for x in (tensor_board + v).tolist())
        s.add(b)

    s2 = set()
    for m in t.expected_moves:
        (b, _, _) = bck.next((board, player_1, t.roll), m)
        key = tuple(b)
        assert key in s
        s2.add(key)

    for x in s:
        assert x in s2
