import backgammon
import pytest

from collections import namedtuple

Light = backgammon.Color.Light
Dark = backgammon.Color.Dark

Case = namedtuple(
    "Case",
    ["start", "moves", "end", "player_1_color", "player", "comment"],
    defaults=[Dark, Dark, ""],
)

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
        moves=[((5, 0), (4, -1), (3, -2), (2, -3))],
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
        player_1_color=Dark,
        player=Light,
        comment="double 5s bearoff",
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
        moves=[((7, 2), (5, 3))],
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
        player_1_color=Dark,
        player=Dark,
        comment="hit on pip 2",
    ),
    Case(
        start="""___________________________________________
|                  |   |                  |
|13 14 15 16 17 18 |   |19 20 21 22 23 24 |
| ○           ●    |   | ●              ○ |
| ○           ●    |   | ●              ○ |
| ○           ●    |   | ●                |
| ○                |   | ●                |
| ○                |   | ●                |
|                  |BAR|                  |
| ●                |   | ○                |
| ●                |   | ○                |
| ●           ○    |   | ○                |
| ●           ○    |   | ○              ● |
| ●           ○    |   | ○              ● |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
""",
        moves=[((24, 18), (18, 13))],
        end="""___________________________________________
|                  |   |                  |
|13 14 15 16 17 18 |   |19 20 21 22 23 24 |
| ○           ●    |   | ●              ○ |
| ○           ●    |   | ●                |
| 6           ●    |   | ●                |
| ○                |   | ●                |
| ○                |   | ●                |
|                  |BAR|                  |
| ●                |   | ○                |
| ●                |   | ○                |
| ●           ○    |   | ○                |
| ●           ○    |   | ○              ● |
| ●           ○    |   | ○              ● |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
""",
        player_1_color=Light,
        player=Light,
        comment="run it on 65",
    ),
    Case(
        start=backgammon.to_str(backgammon.make_board(), player_1_color=Light),
        moves=[((24, 18), (13, 11)), ((24, 18), (18, 14))],
        end="""___________________________________________
|                  |   |                  |
|13 14 15 16 17 18 |   |19 20 21 22 23 24 |
| ○           ●  ○ | ○ | ●              ○ |
| ○           ●    |   | ●                |
| ○           ●    |   | ●                |
| ○                |   | ●                |
|                  |   | ●                |
|                  |BAR|                  |
| ●                |   | ○                |
| ●                |   | ○                |
| ●           ○    |   | ○                |
| ●           ○    |   | ○                |
| ●  ●        ○    |   | ○              ● |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
""",
        player_1_color=Light,
        player=Light,
    ),
]


@pytest.mark.parametrize("t", cases)
def test(t):
    (start, moves, end, player_1_color, player, comment) = t
    board = backgammon.from_str(start, player_1_color=player_1_color)
    for move in moves:
        backgammon.unchecked_move(board, move, player_1=(player == player_1_color))
        player = Dark if player == Light else Light
    expected = backgammon.from_str(end, player_1_color=player_1_color)
    assert expected == board, comment
