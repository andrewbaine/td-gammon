def _m2(board, result, d1, d2, best=0):
    dominant_pip = None
    for i, pc1 in enumerate(board):
        if pc1 > 0:
            dest1 = i + d1
            if i == 0 and board[dest1] < -1:
                break
            if dominant_pip is None:
                dominant_pip = i
            if (
                (board[dest1] < -1)  # occupied by opponent
                if dest1 < 25
                else (
                    (dest1 == 25 and dominant_pip < 19)  # exact bearoff
                    or (dest1 > 25 and dominant_pip != i)  # over-bearoff
                )
            ):
                continue

            board[i] = pc1 - 1
            if i == dominant_pip and board[i] == 0:
                dominant_pip = None

            is_hit = dest1 < 25 and board[dest1] == -1
            if is_hit:
                board[dest1] = 1
            elif dest1 < 25:
                board[dest1] += 1

            if d1 > best:
                best = d1
                result.add((best, ((25 - i, 25 - i - d1),)))
            j = i if d1 < d2 else (i + 1)
            while j < 26:
                pc2 = board[j]
                if pc2 > 0:
                    if dominant_pip is None:
                        dominant_pip = j
                    dest2 = j + d2
                    # if we can land on this dest1 or bearoff
                    if (dest2 < 25 and board[dest2] > -2) or (
                        dominant_pip is None
                        or (dominant_pip > 18 and (dest2 == 25 or dominant_pip == j))
                    ):
                        best = d1 + d2
                        result.add(
                            (best, ((25 - i, 25 - i - d1), (25 - j, 25 - j - d2)))
                        )
                j += 1
            if is_hit:
                board[dest1] = -1
            elif dest1 < 25:
                board[dest1] -= 1

            board[i] = pc1
            if dominant_pip is None or i < dominant_pip:
                dominant_pip = i
    return best


def _dubs(board, result, d1, best=0):
    dominant_pip = None
    for i, pc1 in enumerate(board):
        if pc1 > 0:
            dest1 = i + d1
            if i == 0 and board[dest1] < -1:
                break
            if dominant_pip is None:
                dominant_pip = i
            if (
                (board[dest1] < -1)  # occupied by opponent
                if dest1 < 25
                else (
                    (dest1 == 25 and dominant_pip < 19)  # exact bearoff
                    or (dest1 > 25 and dominant_pip != i)  # over-bearoff
                )
            ):
                continue
            board[i] = pc1 - 1
            if i == dominant_pip and board[i] == 0:
                dominant_pip = None
            is_hit1 = dest1 < 25 and board[dest1] == -1
            if is_hit1:
                board[dest1] = 1
            elif dest1 < 25:
                board[dest1] += 1
            if d1 > best:
                best = d1
                result.add((best, ((25 - i, 25 - i - d1),)))
            j = i
            while j < 26:
                pc2 = board[j]
                if pc2 > 0:
                    dest2 = j + d1
                    if j == 0 and board[dest2] < -1:
                        break
                    if dominant_pip is None:
                        dominant_pip = j
                    if (
                        (board[dest2] < -1)  # occupied by opponent
                        if dest2 < 25
                        else (
                            (dest2 == 25 and dominant_pip < 19)  # exact bearoff
                            or (dest2 > 25 and dominant_pip != j)  # over-bearoff
                        )
                    ):
                        j += 1
                        continue
                    board[j] = pc2 - 1
                    if j == dominant_pip and board[j] == 0:
                        dominant_pip = None
                    is_hit2 = dest2 < 25 and board[dest2] == -1
                    if is_hit2:
                        board[dest2] = 1
                    elif dest2 < 25:
                        board[dest2] += 1
                    if 2 * d1 > best:
                        best = 2 * d1
                        result.add(
                            (best, ((25 - i, 25 - i - d1), (25 - j, 25 - j - d1)))
                        )
                    k = j
                    while k < 26:
                        pc3 = board[k]
                        if pc3 > 0:
                            dest3 = k + d1
                            if k == 0 and board[dest3] < -1:
                                break
                            if dominant_pip is None:
                                dominant_pip = k
                            if (
                                (board[dest3] < -1)  # occupied by opponent
                                if dest3 < 25
                                else (
                                    (dest3 == 25 and dominant_pip < 19)  # exact bearoff
                                    or (
                                        dest3 > 25 and dominant_pip != k
                                    )  # over-bearoff
                                )
                            ):
                                k += 1
                                continue
                            board[k] = pc3 - 1
                            if k == dominant_pip and board[k] == 0:
                                dominant_pip = None
                            is_hit3 = dest3 < 25 and board[dest3] == -1
                            if is_hit3:
                                board[dest3] = 1
                            elif dest3 < 25:
                                board[dest3] += 1
                            if 3 * d1 > best:
                                best = 3 * d1
                                result.add(
                                    (
                                        best,
                                        (
                                            (25 - i, 25 - i - d1),
                                            (25 - j, 25 - j - d1),
                                            (25 - k, 25 - k - d1),
                                        ),
                                    )
                                )
                            l = k
                            while l < 26:
                                pc4 = board[l]
                                if pc4 > 0:
                                    if dominant_pip is None:
                                        dominant_pip = l
                                    dest4 = l + d1
                                    # if we can land on this dest1 or bearoff
                                    if (dest4 < 25 and board[dest4] > -2) or (
                                        dominant_pip is None
                                        or (
                                            dominant_pip > 17
                                            and (dest4 == 25 or dominant_pip == l)
                                        )
                                    ):
                                        best = 4 * d1
                                        result.add(
                                            (
                                                best,
                                                (
                                                    (25 - i, 25 - i - d1),
                                                    (25 - j, 25 - j - d1),
                                                    (25 - k, 25 - k - d1),
                                                    (25 - l, 25 - l - d1),
                                                ),
                                            )
                                        )
                                l += 1
                            if is_hit3:
                                board[dest3] = -1
                            elif dest3 < 25:
                                board[dest3] -= 1
                            board[k] = pc3
                            if dominant_pip is None or k < dominant_pip:
                                dominant_pip = k
                        k += 1
                    if is_hit2:
                        board[dest2] = -1
                    elif dest2 < 25:
                        board[dest2] -= 1
                    board[j] = pc2
                    if dominant_pip is None or j < dominant_pip:
                        dominant_pip = j
                j += 1
            if is_hit1:
                board[dest1] = -1
            elif dest1 < 25:
                board[dest1] -= 1
            board[i] = pc1
            if dominant_pip is None or i < dominant_pip:
                dominant_pip = i
    return best


class MoveComputer:
    def __init__(self):
        self.board = [0 for _ in range(26)]

    def compute_moves(self, board, roll, player_1=True):
        (d1, d2) = roll
        if player_1:
            for i, x in enumerate(board):
                self.board[25 - i] = x
        else:
            for i, x in enumerate(board):
                self.board[i] = -1 * x
        result = set()
        best = 0
        if d1 != d2:
            best = _m2(self.board, result, d1, d2, best)
            best = _m2(self.board, result, d2, d1, best=best)
        else:
            best = _dubs(self.board, result, d1)
        return [x for x in reversed(sorted([b for (a, b) in result if a == best]))]
