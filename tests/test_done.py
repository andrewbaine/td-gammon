from collections import namedtuple
import random

import pytest
import torch

import backgammon
import backgammon_env
import done_check
import slow_but_right

Case = namedtuple(
    "Case", ["board", "player_1_color", "player", "done", "comment"], defaults=[""]
)

Dark = backgammon.Color.Dark
Light = backgammon.Color.Light


cases = [
    Case(
        board="""___________________________________________
|                  |   |                  |
|13 14 15 16 17 18 |   |19 20 21 22 23 24 |
|                  |   | ●  ●     ●  ●  ● |
|                  |   | ●           ●  ● |
|                  |   |             ●  ● |
|                  |   |             ●    |
|                  |   |                  |
|                  |BAR|                  |
|                  |   |                  |
|                  |   |                  |
|                  |   |                  |
|                  |   |                  |
|                  |   |    ●             |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
 """,
        player_1_color=Light,
        player=Dark,
        done=1,
    ),
    Case(
        board="""___________________________________________
|                  |   |                  |
|13 14 15 16 17 18 |   |19 20 21 22 23 24 |
|    ○             |   |                  |
|                  |   |                  |
|                  |   |                  |
|                  |   |                  |
|                  |   |                  |
|                  |BAR|                  |
|                  |   |                  |
|                  |   |          ○  ○    |
|                  |   |          ○  ○  ○ |
|                ○ |   |          ○  ○  ○ |
| ○              ○ |   |          ○  ○  ○ |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
""",
        player_1_color=Light,
        player=Light,
        done=-2,
    ),
]


@pytest.fixture
def bck():
    return backgammon_env.Backgammon(lambda: random.randint(1, 6))


@pytest.fixture
def sbr():
    return slow_but_right.MoveComputer()


@pytest.fixture
def dc():
    return done_check.Donecheck()


@pytest.mark.parametrize("t", cases)
def test(bck, sbr, dc, t):
    player_1 = t.player_1_color == t.player
    board = backgammon.from_str(t.board, player_1_color=t.player_1_color)
    tensor_board = torch.tensor(board, dtype=torch.float)
    state = (board, player_1, (0, 0))
    done = bck.done(state)
    done_slow_but_right = sbr.done(state)
    dc_done = dc.check(tensor_board)
    assert done == t.done
    assert done_slow_but_right == t.done
    assert dc_done == t.done
