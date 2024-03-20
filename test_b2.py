import b2
import backgammon
import pytest

move_computer = b2.MoveComputer()

from collections import namedtuple

Dark = backgammon.Color.Dark
Light = backgammon.Color.Light

Case = namedtuple(
    "TestCase",
    ["board", "roll", "expected_moves", "player_1_color", "player", "comment"],
    defaults=[Dark, Dark, ""],
)

test_cases = [
    Case(
        board="""___________________________________________
|                  |   |                  |
|13 14 15 16 17 18 |   |19 20 21 22 23 24 |
| ●           ○    |   | ○              ● |
| ●           ○    |   | ○              ● |
| ●           ○    |   | ○                |
| ●                |   | ○                |
| ●                |   | ○                |
|                  |BAR|                  |
| ○                |   | ●                |
| ○                |   | ●                |
| ○           ●    |   | ●                |
| ○           ●    |   | ●              ○ |
| ○           ●    |   | ●              ○ |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
""",
        roll=(6, 5),
        expected_moves=[
            ((24, 18), (18, 13)),
            ((24, 18), (13, 8)),
            ((24, 18), (8, 3)),
            ((13, 8), (13, 7)),
            ((13, 8), (8, 2)),
            ((13, 7), (8, 3)),
            ((13, 7), (7, 2)),
            ((8, 3), (8, 2)),
        ],
        comment="starting position, rolled 65",
    ),
    Case(
        board="""___________________________________________
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
|                  |   | ●              ● |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
""",
        roll=(6, 1),
        expected_moves=[
            ((6, 5), (5, -1)),
            ((6, 0), (1, 0)),
        ],
    ),
    Case(
        board="""___________________________________________
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
|                  |   | ●  ●           ● |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
""",
        roll=(5, 2),
        expected_moves=[
            ((6, 4), (5, 0)),
            ((6, 1), (5, 3)),
        ],
    ),
    Case(
        board="""___________________________________________
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
|                  |   | ●  ●           ● |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
""",
        roll=(3, 2),
        expected_moves=[
            ((6, 4), (5, 2)),
            ((6, 4), (4, 1)),
            ((6, 3), (5, 3)),
            ((6, 3), (3, 1)),
            ((5, 3), (3, 0)),
            ((5, 2), (2, 0)),
        ],
    ),
    Case(
        board="""___________________________________________
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
|                  |   | ●  ●           ● |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
""",
        roll=(2, 2),
        expected_moves=[
            ((6, 4), (5, 3), (4, 2), (3, 1)),
            ((6, 4), (5, 3), (4, 2), (2, 0)),
        ],
    ),
    Case(
        board="""___________________________________________
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
|                  |   | ●                |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
""",
        roll=(1, 1),
        expected_moves=[((6, 5), (5, 4), (4, 3), (3, 2))],
    ),
    Case(
        board="""___________________________________________
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
|                  |   | ●              ● |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
""",
        roll=(3, 1),
        expected_moves=[((6, 5), (5, 2)), ((6, 3), (3, 2)), ((6, 3), (1, 0))],
    ),
    Case(
        board="""___________________________________________
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
|                  |   | ●              ● |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
""",
        roll=(1, 1),
        expected_moves=[
            ((6, 5), (5, 4), (4, 3), (3, 2)),
            ((6, 5), (5, 4), (4, 3), (1, 0)),
        ],
    ),
    Case(
        board="""___________________________________________
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
|             ●  ● |   | ●                |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
""",
        roll=(6, 5),
        expected_moves=[
            ((8, 3), (7, 1)),
            ((8, 2), (7, 2)),
            ((8, 2), (6, 1)),
            ((7, 1), (6, 1)),
        ],
    ),
    Case(
        board="""___________________________________________
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
|                ● |   |       ●          |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
""",
        roll=(4, 2),
        expected_moves=[
            ((7, 5), (5, 1)),
            ((7, 5), (4, 0)),
            ((7, 3), (4, 2)),
            ((7, 3), (3, 1)),
        ],
    ),
    Case(
        board="""___________________________________________
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
|                ● |   |       ●          |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
""",
        roll=(5, 1),
        expected_moves=[((7, 6), (6, 1)), ((7, 2), (4, 3)), ((7, 2), (2, 1))],
    ),
    Case(
        board="""___________________________________________
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
|                  |   |    ○             |
|                ● |   |    ○  ●          |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
""",
        roll=(2, 4),
        expected_moves=[((7, 3), (4, 2)), ((7, 3), (3, 1))],
    ),
    Case(
        board="""___________________________________________
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
|                  |   |    ○             |
|                ● |   |    ○  ●          |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
""",
        roll=(1, 1),
        expected_moves=[
            ((7, 6), (4, 3), (3, 2), (2, 1)),
            ((4, 3), (3, 2), (2, 1), (1, 0)),
        ],
    ),
    Case(
        board="""___________________________________________
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
|                ● |   |       ●          |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
""",
        roll=(1, 1),
        expected_moves=[
            ((7, 6), (6, 5), (5, 4), (4, 3)),
            ((7, 6), (6, 5), (4, 3), (3, 2)),
            ((7, 6), (4, 3), (3, 2), (2, 1)),
            ((4, 3), (3, 2), (2, 1), (1, 0)),
        ],
    ),
    Case(
        board="""___________________________________________
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
|                ● |   |    ●             |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
""",
        roll=(1, 1),
        expected_moves=[
            ((7, 6), (6, 5), (5, 4), (5, 4)),
            ((7, 6), (6, 5), (5, 4), (4, 3)),
            ((7, 6), (5, 4), (4, 3), (3, 2)),
            ((5, 4), (4, 3), (3, 2), (2, 1)),
        ],
    ),
    Case(
        board="""___________________________________________
|                  |   |                  |
|13 14 15 16 17 18 |   |19 20 21 22 23 24 |
| ○  ●        ●    | ○ | ●     ●        ○ |
| ○  ●        ●    |   | ●     ●          |
| ○           ●    |   | ●                |
| ○                |   |                  |
|                  |   |                  |
|                  |BAR|                  |
|                  |   | ○                |
|                  |   | ○                |
| ●           ○    |   | ○                |
| ●           ○    |   | ○              ● |
| ●        ○  ○    |   | ○              ● |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
""",
        roll=(2, 2),
        expected_moves=[
            ((25, 23), (24, 22), (22, 20), (20, 18)),
            ((25, 23), (24, 22), (22, 20), (13, 11)),
            ((25, 23), (24, 22), (22, 20), (9, 7)),
            ((25, 23), (24, 22), (22, 20), (8, 6)),
            ((25, 23), (24, 22), (22, 20), (6, 4)),
            ((25, 23), (24, 22), (13, 11), (13, 11)),
            ((25, 23), (24, 22), (13, 11), (11, 9)),
            ((25, 23), (24, 22), (13, 11), (9, 7)),
            ((25, 23), (24, 22), (13, 11), (8, 6)),
            ((25, 23), (24, 22), (13, 11), (6, 4)),
            ((25, 23), (24, 22), (9, 7), (8, 6)),
            ((25, 23), (24, 22), (9, 7), (7, 5)),
            ((25, 23), (24, 22), (9, 7), (6, 4)),
            ((25, 23), (24, 22), (8, 6), (8, 6)),
            ((25, 23), (24, 22), (8, 6), (6, 4)),
            ((25, 23), (24, 22), (6, 4), (6, 4)),
            ((25, 23), (24, 22), (6, 4), (4, 2)),
            ((25, 23), (13, 11), (13, 11), (13, 11)),
            ((25, 23), (13, 11), (13, 11), (11, 9)),
            ((25, 23), (13, 11), (13, 11), (9, 7)),
            ((25, 23), (13, 11), (13, 11), (8, 6)),
            ((25, 23), (13, 11), (13, 11), (6, 4)),
            ((25, 23), (13, 11), (11, 9), (9, 7)),
            ((25, 23), (13, 11), (11, 9), (8, 6)),
            ((25, 23), (13, 11), (11, 9), (6, 4)),
            ((25, 23), (13, 11), (9, 7), (8, 6)),
            ((25, 23), (13, 11), (9, 7), (7, 5)),
            ((25, 23), (13, 11), (9, 7), (6, 4)),
            ((25, 23), (13, 11), (8, 6), (8, 6)),
            ((25, 23), (13, 11), (8, 6), (6, 4)),
            ((25, 23), (13, 11), (6, 4), (6, 4)),
            ((25, 23), (13, 11), (6, 4), (4, 2)),
            ((25, 23), (9, 7), (8, 6), (8, 6)),
            ((25, 23), (9, 7), (8, 6), (7, 5)),
            ((25, 23), (9, 7), (8, 6), (6, 4)),
            ((25, 23), (9, 7), (7, 5), (6, 4)),
            ((25, 23), (9, 7), (7, 5), (5, 3)),
            ((25, 23), (9, 7), (6, 4), (6, 4)),
            ((25, 23), (9, 7), (6, 4), (4, 2)),
            ((25, 23), (8, 6), (8, 6), (8, 6)),
            ((25, 23), (8, 6), (8, 6), (6, 4)),
            ((25, 23), (8, 6), (6, 4), (6, 4)),
            ((25, 23), (8, 6), (6, 4), (4, 2)),
            ((25, 23), (6, 4), (6, 4), (6, 4)),
            ((25, 23), (6, 4), (6, 4), (4, 2)),
        ],
        player_1_color=Light,
        player=Light,
    ),
    Case(
        board="""___________________________________________
|                  |   |                  |
|13 14 15 16 17 18 |   |19 20 21 22 23 24 |
| ○           ●    | ○ | ●           ●  ○ |
| ○           ●    |   | ●           ●    |
| ○                |   | ●                |
| ○                |   | ●                |
|                  |   |                  |
|                  |BAR|                  |
| ●                |   | ○                |
| ●           ○    |   | ○                |
| ●           ○    |   | ○                |
| ●           ○    |   | ○              ● |
| ●           ○    |   | ○              ● |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
""",
        roll=(5, 4),
        expected_moves=[
            ((25, 21), (21, 16)),
            ((25, 21), (13, 8)),
            ((25, 21), (8, 3)),
            ((25, 20), (24, 20)),
            ((25, 20), (20, 16)),
            ((25, 20), (13, 9)),
            ((25, 20), (8, 4)),
            ((25, 20), (6, 2)),
        ],
        player_1_color=Light,
        player=Light,
    ),
]


@pytest.mark.parametrize("t", test_cases)
def tests(t):
    board = backgammon.from_str(t.board, player_1_color=t.player_1_color)
    moves = move_computer.compute_moves(
        board, t.roll, player_1=(t.player == t.player_1_color)
    )
    assert moves == t.expected_moves, t.comment
