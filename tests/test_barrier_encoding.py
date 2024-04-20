import pytest

from collections import namedtuple

import torch
import barrier_encoding
import backgammon

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
]


@pytest.fixture
def barrier_encoder():
    return barrier_encoding.Barrier()


@pytest.fixture
def greatest_barrier_encoder():
    return barrier_encoding.GreatestBarrier()


@pytest.mark.parametrize("t", cases)
def test_barrier_encoding(greatest_barrier_encoder, barrier_encoder, t):
    x = torch.tensor([t.input])
    y = barrier_encoder(x)
    assert y.tolist() == [t.expected]
    y = greatest_barrier_encoder(y)
    assert y.tolist() == [[float(x) for x in t.greatest_barrier_encoding]]
