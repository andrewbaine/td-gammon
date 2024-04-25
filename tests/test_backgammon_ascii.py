from td_gammon import backgammon


def assert_equivalence(board, str, player_1_color=backgammon.Color.Light):
    assert backgammon.to_str(board, player_1_color=player_1_color) == str
    assert backgammon.from_str(str, player_1_color=player_1_color) == board


def test_starting_position_light():
    board = backgammon.make_board()

    str = """___________________________________________
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
"""
    assert_equivalence(board, str, player_1_color=backgammon.Color.Light)


def test_starting_position_default():
    board = backgammon.make_board()
    str = """___________________________________________
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
"""
    assert backgammon.to_str(board) == str
    assert backgammon.from_str(str) == board


def test_starting_position_dark():
    board = backgammon.make_board()
    str = """___________________________________________
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
"""
    assert_equivalence(board, str, player_1_color=backgammon.Color.Dark)


def test_big_stack_from_str():
    str = """___________________________________________
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
"""
    board = backgammon.make_board()
    backgammon.unchecked_move(board, [(24, 18), (18, 13)], player_1=True)
    assert_equivalence(board, str)


def test_hit_from_str():
    board = backgammon.make_board()
    backgammon.unchecked_move(board, [(24, 18), (13, 11)], player_1=True)
    backgammon.unchecked_move(board, [(24, 18), (18, 14)], player_1=False)
    str = """___________________________________________
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
"""
    assert_equivalence(board, str)


def test_weird_board():
    str = """___________________________________________
|                  |   |                  |
|13 14 15 16 17 18 |   |19 20 21 22 23 24 |
| ○  ○        ●    |   |                  |
| ○           ●    |   |                  |
|10           ●    |   |                  |
| ○                |   |                  |
| ○                |   |                  |
|                  |BAR|                  |
|                  | ● |                  |
|                  | ● | ○                |
|                  |11 | ○                |
|             ○    | ● | ○                |
|             ○    | ● | ○           ●  ● |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
"""
    board = [
        -11,
        -1,
        -1,
        0,
        0,
        0,
        4,
        0,
        2,
        0,
        0,
        0,
        0,
        10,
        1,
        0,
        0,
        -3,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert_equivalence(board, str)
