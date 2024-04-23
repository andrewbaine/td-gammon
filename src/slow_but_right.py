class MoveComputer:
    def __init__(self):
        self.scratch = [0 for _ in range(26)]

    def done(self, state):
        """
        At the end of the game, if the losing player has borne off at least one checker,
        he loses only the value showing on the doubling cube (one point, if there have
        been no doubles). However, if the loser has not borne off any of his checkers,
        he is gammoned and loses twice the value of the doubling cube. Or, worse, if the
        loser has not borne off any of his checkers and still has a checker on the bar
        or in the winner's home board, he is backgammoned and loses three times the
        value of the doubling cube.
        """
        (board, player_1, (d1, d2)) = state
        # did player_1 lose?
        player_1_in_back = 0
        player_2_in_back = 0
        player_1_count = 0
        player_2_count = 0
        for i, n in enumerate(board):
            if n < 0:
                if i < 7:
                    player_2_in_back -= n
                player_2_count -= n
            else:
                if i > 18:
                    player_1_in_back += n
                player_1_count += n

        assert 0 <= player_1_count < 16
        assert 0 <= player_2_count < 16
        assert 0 <= player_1_in_back <= player_1_count
        assert 0 <= player_2_in_back <= player_2_count

        if player_2_count == 0:
            assert player_1_count != 0
            if player_1_count == 15:
                if player_1_in_back > 0:
                    return -3
                return -2
            return -1
        if player_1_count == 0:
            assert player_2_count != 0
            if player_2_count == 15:
                if player_2_in_back > 0:
                    return 3
                return 2
            return 1
        return 0

    def play_move(self, start, end):
        board = self.scratch
        assert start > end
        if board[start] <= 0:
            return False
        if end < 0:
            # over-bearoff
            for i in range(25, start, -1):
                if board[i] > 0:
                    return False
            board[start] -= 1
            return True
        elif end == 0:
            for i in range(25, 6, -1):
                if board[i] > 0:
                    return False
            board[start] -= 1
            return True
        else:
            assert end > 0
            if board[25] > 0 and start != 25:
                return False
            if board[end] < -1:
                return False
            if board[end] == -1:
                board[start] -= 1
                board[end] = 1
                board[0] -= 1
                return True
            else:
                assert board[end] >= 0
                board[start] -= 1
                board[end] += 1
                return True

    def reset_board(self, state):
        (board, player_1, (d1, d2)) = state
        if player_1:
            for i, x in enumerate(board):
                self.scratch[i] = x
        else:
            for i, x in enumerate(board):
                self.scratch[25 - i] = -x

    def a_b(self, state):
        moves = []
        (board, player_1, (d1, d2)) = state
        for d1, d2 in [(d1, d2), (d2, d1)]:
            for i in range(25, 0, -1):
                s1 = i
                e1 = i - d1
                for j in range(i - 1 if d1 > d2 else i, 0, -1):
                    s2 = j
                    e2 = j - d2
                    self.reset_board(state)
                    if self.play_move(s1, e1):
                        if self.play_move(s2, e2):
                            moves.append([(s1, e1), (s2, e2)])
        return moves

    def a(self, state):
        (board, player_1, (d1, d2)) = state
        for d in [d1, d2] if d1 > d2 else [d2, d1]:
            moves = []
            for i in range(25, 0, -1):
                start = i
                end = i - d
                self.reset_board(state)
                if self.play_move(start, end):
                    moves.append([(start, end)])
            if moves:
                return moves
        return [()]

    def doubles(self, state):
        (board, player_1, (d1, d2)) = state
        assert d1 == d2
        moves = [[] for _ in range(4)]
        for i in range(25, 0, -1):
            s1 = i
            e1 = i - d1
            for j in range(i, 0, -1):
                s2 = j
                e2 = j - d1
                for k in range(j, 0, -1):
                    s3 = k
                    e3 = k - d1
                    for l in range(k, 0, -1):
                        s4 = l
                        e4 = l - d1
                        self.reset_board(state)
                        if self.play_move(s1, e1):
                            if s2 == s1 and s3 == s2 and s4 == s3:
                                moves[3].append([(s1, e1)])
                            if self.play_move(s2, e2):
                                if s3 == s2 and s4 == s3:
                                    moves[2].append([(s1, e1), (s2, e2)])
                                if self.play_move(s3, e3):
                                    if s4 == s3:
                                        moves[1].append([(s1, e1), (s2, e2), (s3, e3)])
                                    if self.play_move(s4, e4):
                                        moves[0].append(
                                            [(s1, e1), (s2, e2), (s3, e3), (s4, e4)]
                                        )
        for m in moves:
            if m:
                return m
        return [()]

    def compute_moves(self, state):
        (board, player_1, (d1, d2)) = state
        if d1 != d2:
            moves = self.a_b(state)
            if moves:
                return moves
            return self.a(state)
        else:
            return self.doubles(state)


def tesauro_encode(state):
    (board, player_1, _) = state
    xs = [0.0 for _ in range(198)]
    xs[0] = board[0] / -2.0
    for i, x in enumerate(board[1:25]):
        j = 1 + 8 * i
        if x == 1:
            xs[j] = 1.0
        elif x == -1:
            xs[j + 1] = 1.0
        elif x == 2:
            xs[j + 2] = 1.0
        elif x == -2:
            xs[j + 3] = 1.0
        elif x == 3:
            xs[j + 4] = 1.0
        elif x == -3:
            xs[j + 5] = 1.0
        elif x > 3:
            xs[j + 6] = (x - 3) / 2.0
        elif x < -3:
            xs[j + 7] = (x + 3) / -2.0
        else:
            assert x == 0
    xs[1 + 8 * 24] = board[25] / 2.0
    xs[2 + 8 * 24] = sum(x for x in board if x > 0) / 15.0
    xs[3 + 8 * 24] = sum(x for x in board if x < 0) / -15.0
    xs[4 + 8 * 24] = 1.0 if player_1 else 0
    xs[5 + 8 * 24] = 0 if player_1 else 1
    return xs


from typing import List


def simple_baine_encoding_step_2(board: List[int], min, max):
    assert len(board) == 24
    xs = []
    for x in board:
        for y in range(min, max):
            xs.append(1 if x == y else 0)
            xs.append(1 if x == -y else 0)
        xs.append((x - (max - 1)) if x >= max else 0)
        xs.append((-x - (max - 1)) if x <= -max else 0)
    assert len(xs) == (max - min + 1) * 24 * 2
    return xs


def simple_baine_encoding_step_1(board: List[int]):
    bs = []
    for i in range(24):
        x = board[i + 1]
        # y1 is the checker count on the adjacent point to the left
        y1 = 0 if i == 0 else board[i]
        # y2 is the checker count on the the adjacent point to the right
        y2 = 0 if i == 23 else board[i + 2]
        if x > 1 and y1 <= 1:
            c = 0
            j = i + 1
            while j < 25 and board[j] > 1:
                c += 1
                j += 1
            bs.append(c)
        elif x < -1 and y2 >= -1:
            c = 0
            j = i + 1
            while j > 0 and board[j] < -1:
                c -= 1
                j -= 1
            bs.append(c)
        else:
            bs.append(0)
    return bs


def simple_baine_encoding(state):
    (board, _, _) = state
    t = simple_baine_encoding_step_2(simple_baine_encoding_step_1(board), min=1, max=4)
    w = tesauro_encode(state)
    assert len(t) == 192, len(t)
    assert len(w) == 198, len(t)
    print(w[-6:])
    return w + t
