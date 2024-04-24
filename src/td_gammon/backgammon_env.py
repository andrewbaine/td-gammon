from typing import List, Tuple, Union, Set
import random

from . import backgammon

State = Tuple[List[int], bool, Tuple[int, int]]

Movepart = Tuple[int, int]

Move = Union[
    Tuple[()],
    Tuple[Movepart],
    Tuple[Movepart, Movepart],
    Tuple[Movepart, Movepart, Movepart],
    Tuple[Movepart, Movepart, Movepart],
]


class Backgammon:
    def __init__(self, roll=lambda: random.randint(1, 6)):
        self.mc = MoveComputer()
        self.board = tuple(backgammon.make_board())
        self.roll = roll

    def s0(self) -> State:
        dice = None
        while True:
            dice = (self.roll(), self.roll())
            (d1, d2) = dice
            if d1 != d2:
                break
        player_1 = d1 > d2
        return (list(self.board), player_1, dice)

    def available_moves(self, state) -> List[Move]:
        return self.mc.compute_moves(state)

    def next(self, state: State, action: Move) -> State:
        (board, player_1, dice) = state
        dice = (self.roll(), self.roll())
        if action:
            scratch = [x for x in board]
            backgammon.unchecked_move(scratch, action, player_1=player_1)
            return (scratch, not player_1, dice)
        else:
            return (board, not player_1, dice)

    def done(self, state: State):
        (board, player_1, dice) = state
        my_checker_count = 0
        backgammoned = 0
        if player_1:
            for i, x in enumerate(board):
                if x < 0:  # i didnt lose
                    return 0
                if i > 18:
                    backgammoned += x
                my_checker_count += x
            return -1 if my_checker_count < 15 else (-3 if backgammoned else -2)
        else:
            for i, x in enumerate(board):
                if x > 0:  # i didnt lose
                    return 0
                if i < 7:
                    backgammoned -= x
                my_checker_count -= x
            return 1 if my_checker_count < 15 else (3 if backgammoned else 2)


def _m2(board, result, d1, d2, best=0):
    dominant_pip = None
    for i, pc1 in enumerate(board):
        if pc1 > 0:
            dest1 = i + d1
            if dominant_pip is None:
                dominant_pip = i
            if not (
                (board[dest1] < -1)
                if (dest1 < 25)
                else ((dominant_pip < 19) if dest1 == 25 else (dominant_pip != i))
            ):
                board[i] = pc1 - 1
                if i == dominant_pip and board[i] == 0:
                    dominant_pip = None

                is_hit = dest1 < 25 and board[dest1] == -1
                if is_hit:
                    board[dest1] = 1
                elif dest1 < 25:
                    board[dest1] += 1

                if d1 >= best:
                    best = d1
                    result.add((best, ((25 - i, 25 - i - d1),)))
                for j in range(i, 26):
                    pc2 = board[j]
                    if pc2 > 0:
                        if dominant_pip is None:
                            dominant_pip = j
                        dest2 = j + d2
                        # if we can land on this dest1 or bearoff
                        if not (
                            (board[dest2] < -1)
                            if (dest2 < 25)
                            else (
                                (dominant_pip < 19)
                                if dest2 == 25
                                else (dominant_pip != j)
                            )
                        ):
                            best = d1 + d2
                            if i == j:
                                result.add(
                                    (
                                        best,
                                        (
                                            (25 - i, 25 - i - (d1 if d1 < d2 else d2)),
                                            (25 - j, 25 - j - (d2 if d1 < d2 else d1)),
                                        ),
                                    )
                                )
                            else:
                                result.add(
                                    (
                                        best,
                                        ((25 - i, 25 - i - d1), (25 - j, 25 - j - d2)),
                                    )
                                )

                        if j == 0:
                            break
                if is_hit:
                    board[dest1] = -1
                elif dest1 < 25:
                    board[dest1] -= 1

                board[i] = pc1
                if dominant_pip is None or i < dominant_pip:
                    dominant_pip = i
            if i == 0:
                break
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
                (board[dest1] < -1)
                if dest1 < 25
                else (dominant_pip < 19 if dest1 == 25 else dominant_pip != i)
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
            for j in range(i, 26):
                pc2 = board[j]
                if pc2 > 0:
                    dest2 = j + d1
                    if j == 0 and board[dest2] < -1:
                        break
                    if dominant_pip is None:
                        dominant_pip = j
                    if (
                        (board[dest2] < -1)
                        if dest2 < 25
                        else (dominant_pip < 19) if dest2 == 25 else (dominant_pip != j)
                    ):
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
                    for k in range(j, 26):
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
                                    (dominant_pip < 19)  # bearoff before bearin
                                    if dest3 == 25
                                    else dominant_pip
                                    != k  # over-bearoff when not the dominant pip
                                )
                            ):
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
                            for l in range(k, 26):
                                pc4 = board[l]
                                if pc4 > 0:
                                    if dominant_pip is None:
                                        dominant_pip = l
                                    dest4 = l + d1
                                    if (
                                        (board[dest4] > -2)
                                        if dest4 < 25
                                        else (
                                            (dominant_pip > 18)
                                            if dest4 == 25
                                            else (dominant_pip == l)
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
                                    if l == 0:
                                        break
                            if is_hit3:
                                board[dest3] = -1
                            elif dest3 < 25:
                                board[dest3] -= 1
                            board[k] = pc3
                            if dominant_pip is None or k < dominant_pip:
                                dominant_pip = k
                            if k == 0:
                                break
                    if is_hit2:
                        board[dest2] = -1
                    elif dest2 < 25:
                        board[dest2] -= 1
                    board[j] = pc2
                    if dominant_pip is None or j < dominant_pip:
                        dominant_pip = j
                    if j == 0:
                        break
            if is_hit1:
                board[dest1] = -1
            elif dest1 < 25:
                board[dest1] -= 1
            board[i] = pc1
            if dominant_pip is None or i < dominant_pip:
                dominant_pip = i
            if i == 0:
                break
    return best


class MoveComputer:
    def __init__(self):
        self.board = [0 for _ in range(26)]

    def compute_moves(self, state) -> List[Move]:
        (board, player_1, (d1, d2)) = state
        if player_1:
            for i, x in enumerate(board):
                self.board[25 - i] = x
        else:
            for i, x in enumerate(board):
                self.board[i] = -1 * x
        result: Set[Tuple[int, Move]] = set()
        result.add((0, ()))
        best = 0
        if d1 != d2:
            best = _m2(self.board, result, d1, d2, best)
            best = _m2(self.board, result, d2, d1, best=best)
        else:
            best = _dubs(self.board, result, d1)
        return [x for x in reversed(sorted([b for (a, b) in result if a == best]))]
