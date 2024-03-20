from enum import Enum


class Color(Enum):
    Dark = 0
    Light = 1


def make_board():
    return [
        0,
        -2,
        0,
        0,
        0,
        0,
        5,
        0,
        3,
        0,
        0,
        0,
        -5,
        5,
        0,
        0,
        0,
        -3,
        0,
        -5,
        0,
        0,
        0,
        0,
        2,
        0,
    ]


def unchecked_move(board, move, player_1=True):
    if player_1:
        for src, dest in move:
            board[src] = board[src] - 1
            if dest > 0:
                if board[dest] == -1:
                    board[0] -= 1
                    board[dest] = 1
                else:
                    board[dest] += 1
    else:
        for src, dest in move:
            board[25 - src] = board[25 - src] + 1
            if dest > 0:
                if board[25 - dest] == 1:
                    board[25] += 1
                    board[25 - dest] = -1
                else:
                    board[25 - dest] -= 1


def s(count, n, checkers):
    if n < 1 or n > 5:
        raise Exception("shouldnt happen")
    abs_count = abs(count)
    if abs_count > 5 and n == 3:
        return str(abs_count).rjust(2) + " "
    if abs_count < n:
        return "   "
    (player_1_checker, player_2_checker) = checkers
    return player_2_checker if count < 0 else player_1_checker


def to_str(board, player_1_color=Color.Light):
    lines = []
    lines.append("___________________________________________")
    lines.append("|                  |   |                  |")
    lines.append("|13 14 15 16 17 18 |   |19 20 21 22 23 24 |")

    checkers = (" ● ", " ○ ") if player_1_color == Color.Dark else (" ○ ", " ● ")

    for i in range(1, 6):
        line = ["|"]
        for j in range(13, 19):
            count = board[j]
            line.append(s(count, i, checkers))
        line.append("|")
        count = board[25]
        line.append(s(count, i, checkers))
        line.append("|")
        for j in range(19, 25):
            count = board[j]
            line.append(s(count, i, checkers))
        line.append("|")
        lines.append("".join(line))
    lines.append("|                  |BAR|                  |")
    for i in range(5, 0, -1):
        line = ["|"]
        for j in range(12, 6, -1):
            count = board[j]
            line.append(s(count, i, checkers))
        line.append("|")
        count = board[0]
        line.append(s(count, i, checkers))
        line.append("|")
        for j in range(6, 0, -1):
            count = board[j]
            line.append(s(count, i, checkers))
        line.append("|")
        lines.append("".join(line))
    lines.append("|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |")
    lines.append("|__________________|___|__________________|")
    lines.append("")
    return "\n".join(lines)


def __update_board(line, s, board, j, player_1_color):
    current_count = board[j]
    token = line[s : s + 3]
    match (token, player_1_color):
        case ("   ", _):
            pass
        case (" ● ", Color.Dark) | (" ○ ", Color.Light):
            board[j] = current_count + 1 if current_count < 5 else current_count
        case (" ● ", Color.Light) | (" ○ ", Color.Dark):
            board[j] = current_count - 1 if current_count > -5 else current_count
        case (
            (" 6 ", _)
            | (" 7 ", _)
            | (" 8 ", _)
            | (" 9 ", _)
            | ("10 ", _)
            | ("11 ", _)
            | ("12 ", _)
            | ("13 ", _)
            | ("14 ", _)
            | ("15 ", _)
            | ("16 ", _)
        ):
            n = int(token)
            if current_count == 2:
                board[j] = n
            elif current_count == -2:
                board[j] = -1 * n
            else:
                raise Exception(token)
        case x:
            raise Exception(x)


def from_str(ascii, player_1_color=Color.Light):
    board = [0 for _ in range(0, 26)]
    lines = ascii.split("\n")

    if lines[0] != "___________________________________________":
        raise Exception("bad first line")
    if lines[1] != "|                  |   |                  |":
        raise Exception("bad second line")
    if lines[2] != "|13 14 15 16 17 18 |   |19 20 21 22 23 24 |":
        raise Exception("bad third line")
    for i in range(1, 6):
        line = lines[i + 2]
        for j in range(13, 19):
            s = 1 + (j - 13) * 3
            __update_board(line, s, board, j, player_1_color)
        for j in range(19, 25):
            s = 24 + (j - 19) * 3
            __update_board(line, s, board, j, player_1_color)
        j = 25
        s = 20
        __update_board(line, s, board, j, player_1_color)
        line = lines[-3 - i]
        for j in range(12, 6, -1):
            s = 1 + (12 - j) * 3
            __update_board(line, s, board, j, player_1_color)
        for j in range(6, 0, -1):
            s = 24 + (6 - j) * 3
            __update_board(line, s, board, j, player_1_color)
        j = 0
        s = 20
        __update_board(line, s, board, j, player_1_color)
    return board


def invert(board):
    i = 0
    while i < 13:
        j = 25 - i
        (board[i], board[j]) = (-1 * board[j], -1 * board[i])
        i += 1
