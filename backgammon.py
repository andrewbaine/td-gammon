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

def checked_move(board, player, move):
    bar = 25 if player else 0
    other_bar = 0 if player else 25
    for (src, dest) in move:
        if board[bar] > 0 and src != bar:
            raise Exception("must reenter")
        src_occupied = board[src] > 0 if player else board[src] < 0
        if src_occupied > 0:
            is_bearoff = dest < 1 if player else dest > 24
            if is_bearoff:
                # TODO (baine): check for bearoff allowed
                board[src] = board[src] - (1 if player else -1)
                return
            if board[dest] < -1 if player else board[dest] > 1:
                raise Exception("cant move to someone else's point")
            elif board[dest] == (-1 if player else 1):
                board[src] = board[src] - (1 if player else -1)
                board[other_bar] = board[other_bar] + (-1 if player else 1)
                board[dest] = 1 if player else -1
            else:
                board[src] -= (1 if player else -1)
                board[dest] += (1 if player else -1)
        else:
            raise Exception("no piece on source")

def allowed_moves(board, roll):
    (d1, d2) = roll
    if d1 == d2:
        raise Exception("we dont do doubles")
    if d1 < d2:
        return allowed_moves(board, (d2, d1))
    if board[25] > 1:
        # we need to use both moves to reenter
        match (board[25- d1], board[25 - d2]):
            case (True, True):
                return [[(25, 25 - d1), (25, 25 - d2)]]
            case (True, False):
                return [[(25, 25 - d1)]]
            case (False, True):
                return [[(25, 25 - d2)]]
            case (False, False):
                return []
    if board[25] == 1:
        single_moves = []
        double_moves = []
        for (m1, m2) in [(d1, d2), (d2, d1)]:
            # can we re-enter with m1? no need to test m2, we'll do so after
            if board[25 - m1] > -2:
                reentry_move = (25, 25 - m1)
                if board[25 - m1] < 1 and board[25 - m1 - m2] > -2:
                    double_moves.append([reentry_move, (25 - m1, 25 - m2)])
                i = 24
                # if we can re-enter, are there other moves available?
                while i > m2:
                    if board[i] > 0 and board[i - m2] > -2:
                        double_moves.append([reentry_move, (i, i - m2)])
                        i -= 1
                if not double_moves and not single_moves:
                    # it's important that d1 > d2 so we use the
                    # higher die move only if both are possibel
                    single_moves.append([(25, 25 - m1)])
        return double_moves or single_moves

    double_moves = []
    bearoff_possible = True
    s1 = []
    s2 = []
    for (m1, m2, single_moves) in [(d1, d2, s1), (d2, d1, s2)]:
        i = 24
        while i > 0:
            if board[i] > 0:
                if i > m1:
                    if board[i - m1] > -1:
                        single_moves.append([(i, i - m1)])
                        board[i] -= 1
                        previous_count = board[i - m1]
                        board[i - m1] = max(1, (previous_count + 1))
                        j = i if m1 > m2 else i - 1
                        pbp = bearoff_possible
                        while j > 0:
                            if board[j] > 0:
                                if j > m2:
                                    if board[j - m2] > -1:
                                        double_moves.append([(i, i - m1), (j, j - m2)])
                                elif bearoff_possible:
                                    double_moves.append([(i, i - m1), (j, 0)])
                                bearoff_possible = bearoff_possible and j < 7
                            j = j - 1
                        bearoff_possible = pbp
                        board[i - m1] = previous_count
                        board[i] += 1
                    bearoff_possible = bearoff_possible and i < 7
                elif bearoff_possible:
                    single_moves.append([(i, i - m1)])
                    board[i] -= 1
                    j = i if m1 > m2 else i - 1
                    while j > 0:
                        if board[j] > 0:
                            if j > m2:
                                if board[j - m2] > -1:
                                    double_moves.append([(i, i - m1), (j, j - m2)])
                            else:
                                double_moves.append([(i, i - m1), (j, j - m2)])
                        j -= 1
                    board[i] += 1
            i = i - 1
    return double_moves or s1 or s2


def s(count, n):
    if n < 1 or n > 5:
        raise Exception("shouldnt happen")
    dark_checker =  " ● "
    light_checker = " ○ "
    abs_count = abs(count)
    if abs_count > 5 and n == 3:
        return str(abs_count).rjust(2) + " "
    else:
        return "   " if abs_count < n else dark_checker if count < 0 else light_checker

def to_ascii(board):
    lines = []
    lines.append("___________________________________________")
    lines.append("|                  |   |                  |")
    lines.append("|13 14 15 16 17 18 |   |19 20 21 22 23 24 |")

    for i in range(1, 6):
        line = ["|"]
        for j in range(13, 19):
            count = board[j]
            line.append(s(count, i))
        line.append("|")
        count = board[25]
        line.append(s(count, i))
        line.append("|")
        for j in range(19, 25):
            count = board[j]
            line.append(s(count, i))
        line.append("|")
        lines.append("".join(line))
    lines.append("|                  |BAR|                  |")
    for i in range(5, 0, -1):
        line = ["|"]
        for j in range(12, 6, -1):
            count = board[j]
            line.append(s(count, i))
        line.append("|")
        count = board[0]
        line.append(s(count, i))
        line.append("|")
        for j in range(6, 0, -1):
            count = board[j]
            line.append(s(count, i))
        line.append("|")
        lines.append("".join(line))
    lines.append("|12 11 10  9  8  7 |   | 6  5  4  3  2  1 |")
    lines.append("|__________________|___|__________________|")
    lines.append("")
    return "\n".join(lines)

def update_board(line, s, board, j):
    current_count = board[j]
    token =  line[s:s + 3]
    match token:
        case "   ":
            pass
        case " ● ":
            board[j] = current_count -1 if current_count > -5 else current_count
        case " ○ ":
            board[j] = current_count + 1 if current_count < 5 else current_count
        case " 6 " | " 7 " | " 8 " | " 9 " | "10 " | "11 " | "12 " | "13 " | "14 " | "15 " | "16 ":
            n = int(token)
            if current_count == 2:
                board[j] = n
            elif current_count == -2:
                board[j] = -1 * n
            else:
                raise Exception(token)
        case x:
            raise Exception(x)
    

def from_ascii(ascii):
    board = [0 for _ in range(0, 26)]
    lines = ascii.split("\n")

    if lines[0] !=     "___________________________________________":
        raise Exception("bad first line")
    if lines[1] != "|                  |   |                  |":
        raise Exception("bad second line")
    if lines[2] != "|13 14 15 16 17 18 |   |19 20 21 22 23 24 |":
        raise Exception("bad third line")
    for i in range(1, 6):
        line = lines[i + 2]
        for j in range(13, 19):
            s = 1 + (j - 13) * 3
            update_board(line, s, board, j)
        for j in range(19, 25):
            s = 24 + (j - 19) * 3
            update_board(line, s, board, j)
        line = lines[-3 - i]
        for j in range(12, 6, -1):
            s = 1 + (12 - j) * 3
            update_board(line, s, board, j)
        for j in range(6, 0, -1):
            s = 24 + (6 - j) * 3
            update_board(line, s, board, j)
    return board
