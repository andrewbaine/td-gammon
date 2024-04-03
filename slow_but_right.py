class MoveComputer:
    def __init__(self):
        self.scratch = [0 for _ in range(26)]

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
        return []

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
        return []

    def compute_moves(self, state):
        (board, player_1, (d1, d2)) = state
        if d1 != d2:
            moves = self.a_b(state)
            if moves:
                return moves
            return self.a(state)
        else:
            return self.doubles(state)
