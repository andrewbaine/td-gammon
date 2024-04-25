import struct
import plyvel
from pytest import approx
import pytest

cases = [
    # https://bkgm.com/articles/Zare/EffectivePipCount/index.html
    ([0, 0, 0, 0, 0, 1], 10.208),
    ([3, 0, 0, 3, 1], 32.425),
    ([0, 0, 0, 3, 5, 7], 86.07),
    ([2, 2, 2, 3, 3, 3], 66.47),
    ([3, 3, 3, 2, 2, 2], 61.71),
    ([0, 0, 0, 1, 1, 1], 20.31),
    ((12,), 43.00),
    ([2, 2], 15.2),
    ([0, 4], 18.476),
    ([4, 4, 4, 3], 57.73),
    ([3, 0, 1, 1], 2.7386 * 49 / 6),
    ([0, 0, 2, 0, 5, 8], 79 + 7.49),
    ([1, 2, 2, 0, 5, 5], 66 + 9.31),
    ([0, 2, 1, 2, 4, 4], 66.59),
    ([0, 3, 1, 2, 3, 4], 64.37),
    ([0, 0, 0, 3, 6, 6], 7.11 + 78),
    ([0, 0, 0, 1, 4, 10], 84 + 7.35),
    ([0, 0, 0, 1, 10, 4], 78 + 8.39),
    ([0, 0, 0, 10, 2, 3], 78.50),
]


@pytest.fixture
def db():
    with plyvel.DB("epc.7.v3.db", create_if_missing=False) as db:
        yield db


@pytest.mark.parametrize("t", cases)
def test_x(db, t):
    (xs, expected) = t
    key = bytes(xs)
    v = db.get(key)
    (v,) = struct.unpack("f", v)
    assert v * 49 / 6 == approx(expected, rel=1e-3), xs
