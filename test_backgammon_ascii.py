
import backgammon

def assert_equivalence(board, ascii, player_1_color=backgammon.Color.Light):
    assert backgammon.to_ascii(board, player_1_color=player_1_color) == ascii
    assert backgammon.from_ascii(ascii, player_1_color=player_1_color) == board

def test_starting_position_light():
    board = backgammon.make_board()

    ascii = """___________________________________________
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
    assert_equivalence(board, ascii, player_1_color=backgammon.Color.Light)

def test_starting_position_default():
    board = backgammon.make_board()
    ascii = """___________________________________________
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
    assert backgammon.to_ascii(board) == ascii
    assert backgammon.from_ascii(ascii) == board

def test_starting_position_dark():
    board = backgammon.make_board()
    ascii = """___________________________________________
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
    assert_equivalence(board, ascii, player_1_color=backgammon.Color.Dark)

def test_big_stack():
    board = backgammon.make_board()
    backgammon.checked_move(board, True, [(24, 18), (18, 13)])
    ascii = """___________________________________________
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
    assert_equivalence(board, ascii, player_1_color=backgammon.Color.Light)


def test_hit():
    board = backgammon.make_board()
    backgammon.checked_move(board, True, [(24, 18), (13, 11)])
    backgammon.checked_move(board, False, [(1, 7), (7, 11)])
    ascii = """___________________________________________
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
    assert_equivalence(board, ascii)



def test_big_stack_from_ascii():
    ascii = """___________________________________________
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
    backgammon.checked_move(board, True, [(24, 18), (18, 13)])
    assert_equivalence(board, ascii)

def test_hit_from_ascii():
    board = backgammon.make_board()
    backgammon.checked_move(board, True, [(24, 18), (13, 11)])
    backgammon.checked_move(board, False, [(1, 7), (7, 11)])
    ascii = """___________________________________________
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
    assert_equivalence(board, ascii)

def test_hit_from_ascii_2():
    """this should test white hitting black"""
    board = backgammon.make_board()
    backgammon.checked_move(board, True, [(8, 5), (6, 5)])
    backgammon.checked_move(board, False, [(1, 2), (12, 14)])
    backgammon.checked_move(board, True, [(24, 18), (18, 14)])
    ascii = """___________________________________________
|                  |   |                  |
|13 14 15 16 17 18 |   |19 20 21 22 23 24 |
| ○  ○        ●    |   | ●              ○ |
| ○           ●    |   | ●                |
| ○           ●    |   | ●                |
| ○                |   | ●                |
| ○                |   | ●                |
|                  |BAR|                  |
|                  |   |                  |
| ●                |   | ○                |
| ●                |   | ○                |
| ●           ○    |   | ○  ○             |
| ●           ○    | ● | ○  ○        ●  ● |
|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |
|__________________|___|__________________|
"""
    assert_equivalence(board, ascii)


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
    board = [-11, -1, -1, 0, 0, 0, 4,
                  0, 2, 0, 0, 0, 0,
                 10, 1, 0, 0, -3, 0,
                  0, 0, 0, 0, 0, 0, 0]
    assert_equivalence(board, str)
