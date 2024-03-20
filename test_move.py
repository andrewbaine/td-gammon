import backgammon
import pytest

from collections import namedtuple

Case = namedtuple("Case", ["start", "move", "end", "player", "comment"], defaults=[""])

cases = [
    Case(
        start="""___________________________________________
|                  |   |                  |
|13 14 15 16 17 18 |   |19 20 21 22 23 24 |
|                  |   |    ○  ○  ○  ○  ○ |
|                  |   |                  |
|                  |   |                  |
|                  |   |                  |
|                  |   |                  |
|                  |BAR|                  |
|                  |   |                  |
|                  |   |                  |
|                  |   |                  |
|                  |   |                  |
|                ● |   |    ●  ●  ●  ●  ● |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
""",
        move=[(5, 0), (4, -1), (3, -2), (2, -3)],
        end="""___________________________________________
|                  |   |                  |
|13 14 15 16 17 18 |   |19 20 21 22 23 24 |
|                  |   |                ○ |
|                  |   |                  |
|                  |   |                  |
|                  |   |                  |
|                  |   |                  |
|                  |BAR|                  |
|                  |   |                  |
|                  |   |                  |
|                  |   |                  |
|                  |   |                  |
|                ● |   |    ●  ●  ●  ●  ● |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
""",
        player=backgammon.Color.Light,
    ),
    Case(
        start="""___________________________________________
|                  |   |                  |
|13 14 15 16 17 18 |   |19 20 21 22 23 24 |
|                  |   |    ○  ○  ○  ○  ○ |
|                  |   |                  |
|                  |   |                  |
|                  |   |                  |
|                  |   |                  |
|                  |BAR|                  |
|                  |   |                  |
|                  |   |                  |
|                  |   |                  |
|                  |   |                ○ |
|                ● |   |    ●  ●  ●  ○  ○ |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
""",
        move=[(7, 2), (5, 3)],
        end="""___________________________________________
|                  |   |                  |
|13 14 15 16 17 18 |   |19 20 21 22 23 24 |
|                  |   |    ○  ○  ○  ○  ○ |
|                  |   |                  |
|                  |   |                  |
|                  |   |                  |
|                  |   |                  |
|                  |BAR|                  |
|                  |   |                  |
|                  |   |                  |
|                  |   |                  |
|                  |   |          ●     ○ |
|                  | ○ |       ●  ●  ●  ○ |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
""",
        player=backgammon.Color.Dark,
        comment="hit on pip 2",
    ),
]


@pytest.mark.parametrize("t", cases)
def test(t):
    (start, move, end, player, comment) = t
    board = backgammon.from_str(start, player_1_color=backgammon.Color.Dark)
    backgammon.unchecked_move(board, move, player_1=(player == backgammon.Color.Dark))
    expected = backgammon.from_str(end, player_1_color=backgammon.Color.Dark)
    assert expected == board, comment
