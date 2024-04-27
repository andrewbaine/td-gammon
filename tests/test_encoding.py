import plyvel
from collections import namedtuple

import pytest
import torch

from td_gammon import backgammon, encoders
from . import slow_but_right

Case = namedtuple("Case", ["input", "expected", "greatest_barrier_encoding"])


def f(x):
    [a, b, c, d] = x
    return a + b + c + d


cases = [
    Case(
        input=f(
            [
                [-2, 0, 0, 0, 0, 5],
                [0, 3, 0, 0, 0, -5],
                [5, 0, 0, 0, -3, 0],
                [-5, 0, 0, 0, 0, 2],
            ]
        ),
        expected=f(
            [
                [-1, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, -1],
                [1, 0, 0, 0, -1, 0],
                [-1, 0, 0, 0, 0, 1],
            ]
        ),
        greatest_barrier_encoding=f(
            [
                [-1, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, -1],
                [1, 0, 0, 0, -1, 0],
                [-1, 0, 0, 0, 0, 1],
            ]
        ),
    ),
    Case(
        input=backgammon.make_board()[1:25],
        expected=f(
            [
                [-1, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, -1],
                [1, 0, 0, 0, -1, 0],
                [-1, 0, 0, 0, 0, 1],
            ]
        ),
        greatest_barrier_encoding=f(
            [
                [-1, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, -1],
                [1, 0, 0, 0, -1, 0],
                [-1, 0, 0, 0, 0, 1],
            ]
        ),
    ),
    Case(
        input=f(
            [
                [3, 3, 3, 3, 3, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, -3, -3, -3, -3, -3],
            ]
        ),
        expected=f(
            [
                [5, 4, 3, 2, 1, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, -1, -2, -3, -4, -5],
            ]
        ),
        greatest_barrier_encoding=f(
            [
                [5, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -5],
            ]
        ),
    ),
    Case(
        input=f(
            [
                [-3, -3, -3, -3, -3, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 3, 3, 3, 3, 3],
            ]
        ),
        expected=f(
            [
                [-1, -2, -3, -4, -5, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 5, 4, 3, 2, 1],
            ]
        ),
        greatest_barrier_encoding=f(
            [
                [0, 0, 0, 0, -5, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 5, 0, 0, 0, 0],
            ]
        ),
    ),
    Case(
        input=f(
            [
                [0, 0, 0, -15, 15, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
        expected=f(
            [
                [0, 0, 0, -1, 1, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
        greatest_barrier_encoding=f(
            [
                [0, 0, 0, -1, 1, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
    ),
    Case(
        input=f(
            [
                [0, 0, 0, 15, -15, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
        expected=f(
            [
                [0, 0, 0, 1, -1, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
        greatest_barrier_encoding=f(
            [
                [0, 0, 0, 1, -1, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
    ),
    Case(
        input=f(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
        expected=f(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
        greatest_barrier_encoding=f(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
    ),
    Case(
        input=f(
            [
                [0, 0, 0, 0, 0, 0],
                [2, 2, -3, -3, 2, 2],
                [-3, -3, 2, 2, -2, -1],
                [2, 1, 0, 0, 0, 0],
            ]
        ),
        expected=f(
            [
                [0, 0, 0, 0, 0, 0],
                [2, 1, -1, -2, 2, 1],
                [-1, -2, 2, 1, -1, 0],
                [1, 0, 0, 0, 0, 0],
            ]
        ),
        greatest_barrier_encoding=f(
            [
                [0, 0, 0, 0, 0, 0],
                [2, 0, 0, -2, 2, 0],
                [0, -2, 2, 0, -1, 0],
                [1, 0, 0, 0, 0, 0],
            ]
        ),
    ),
    Case(
        input=f(
            [
                [0, 0, 0, 0, 0, 0],
                [2, -2, 3, -3, 2, -2],
                [3, -3, 2, -2, 2, -2],
                [1, -1, 0, 0, 0, 0],
            ]
        ),
        expected=f(
            [
                [0, 0, 0, 0, 0, 0],
                [1, -1, 1, -1, 1, -1],
                [1, -1, 1, -1, 1, -1],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
        greatest_barrier_encoding=f(
            [
                [0, 0, 0, 0, 0, 0],
                [1, -1, 1, -1, 1, -1],
                [1, -1, 1, -1, 1, -1],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
    ),
]


@pytest.fixture
def barrier_encoder():
    return encoders.Barrier()


@pytest.fixture
def greatest_barrier_encoder():
    return encoders.GreatestBarrier()


@pytest.mark.parametrize("t", cases)
def test_barrier_encoding(greatest_barrier_encoder, barrier_encoder, t):
    x = torch.tensor([t.input])
    y = barrier_encoder(x)
    assert y.tolist() == [[float(x) for x in t.expected]]
    y = greatest_barrier_encoder(y)
    assert y.tolist() == [[float(x) for x in t.greatest_barrier_encoding]]
    yy = slow_but_right.simple_baine_encoding_step_1(t.input)
    assert [yy] == y.tolist()


def test_matrix(greatest_barrier_encoder, barrier_encoder):
    inputs = [c.input for c in cases]
    y = barrier_encoder(torch.tensor(inputs))
    expected = [c.expected for c in cases]
    assert y.tolist() == expected
    y = greatest_barrier_encoder(y)
    assert y.tolist() == [c.greatest_barrier_encoding for c in cases]
    yy = [slow_but_right.simple_baine_encoding_step_1(t.input) for t in cases]
    assert yy == y.tolist()


@pytest.fixture
def db():
    with plyvel.DB("epc.12.v3.db", create_if_missing=False) as db:
        yield db


def test_epc(db):
    board = backgammon.make_board()
    places = [(1, 7), (7, 19), (19, 26)]
    baine_epc = encoders.EPC(db, places)
    t = torch.tensor([board + [1]])
    y = baine_epc(t).tolist()
    assert y == [
        [
            4.51819372177124,
            6.55345344543457,
            2.1096107959747314,
            4.51819372177124,
            6.55345344543457,
            2.1096107959747314,
        ]
    ]
