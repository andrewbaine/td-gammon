import backgammon


# content of test_sample.py
def inc(x):
    return x + 1


def test_answer():
    assert inc(3) == 4


def test_starting_position():
    b = backgammon.make_board()
    ascii = backgammon.to_ascii(b)

    expected_ascii = """___________________________________________
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
    assert ascii == expected_ascii


def test_6_5():
    board = backgammon.make_board()
    expected = [
        [(24, 18), (18, 13)],
        [(24, 18), (13, 8)],
        [(24, 18), (8, 3)],
        [(13, 7), (13, 8)],
        [(13, 7), (8, 3)],
        [(13, 7), (7, 2)],
        [(8, 2), (8, 3)],
        [(13, 8), (8, 2)],
    ]
    for roll in [(6, 5), (5, 6)]:
        assert backgammon.allowed_moves(board, roll) == expected


def test_big_stack():
    b = backgammon.make_board()
    backgammon.checked_move(b, True, [(24, 18), (18, 13)])
    expected_ascii = """___________________________________________
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
    assert backgammon.to_ascii(b) == expected_ascii


def test_hit():
    b = backgammon.make_board()
    backgammon.checked_move(b, True, [(24, 18), (13, 11)])
    backgammon.checked_move(b, False, [(1, 7), (7, 11)])
    expected_ascii = """___________________________________________
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
    assert backgammon.to_ascii(b) == expected_ascii


def test_from_ascii():
    expected = backgammon.make_board()
    ascii = backgammon.to_ascii(expected)
    actual = backgammon.from_ascii(ascii)
    assert actual == expected


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
    actual = backgammon.from_ascii(ascii)
    board = backgammon.make_board()
    backgammon.checked_move(board, True, [(24, 18), (18, 13)])

    assert actual == board
