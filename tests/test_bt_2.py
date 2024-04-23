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
    print(t.board)
    print(t.roll)
    board = backgammon.from_str(t.board, player_1_color=t.player_1_color)
    player_1 = t.player == t.player_1_color
    tensor_board = tensor([board + [1 if player_1 else 0]])
    (vectors) = move_tensors.compute_move_vectors(tensor_board, t.roll)
    s = set()
    for v in (vectors + tensor_board).tolist():
        print(v)
        p = v[-1]
        assert p if not player_1 else (not p)
        b = tuple(int(x) for x in v[:-1])
        s.add(b)

    for x in s:
        print("x", x)

    s2 = set()
    for m in t.expected_moves:
        (b, _, _) = bck.next((board, player_1, t.roll), m)
        key = tuple(b)
        print("key", key)
        assert key in s
        s2.add(key)

    for x in s:
        assert x in s2
