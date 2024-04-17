def f(new_d, dict, board, n):
    key = tuple((i, x) for i, x in enumerate(board) if x > 0)
    if key in new_d:
        return new_d[key]
    if key in dict:
        return dict[key]
    expectation = 0

    for d1 in range(1, 7):
        for d2 in range(1, 7):
            best = float("inf")
            if d1 == d2:
                for s1 in range(0, n):
                    if board[s1] > 0:
                        board[s1] -= 1
                        e1 = s1 - d1
                        if e1 > -1:
                            board[e1] += 1
                        best = min(best, 1 + f(new_d, dict, board, n))
                        for s2 in range(0, n):
                            if board[s2] > 0:
                                board[s2] -= 1
                                e2 = s2 - d1
                                if e2 > -1:
                                    board[e2] += 1
                                best = min(best, 1 + f(new_d, dict, board, n))
                                for s3 in range(0, n):
                                    if board[s3] > 0:
                                        board[s3] -= 1
                                        e3 = s3 - d1
                                        if e3 > -1:
                                            board[e3] += 1
                                        best = min(best, 1 + f(new_d, dict, board, n))
                                        for s4 in range(0, n):
                                            if board[s4] > 0:
                                                board[s4] -= 1
                                                e4 = s4 - d1
                                                if e4 > -1:
                                                    board[e4] += 1
                                                best = min(
                                                    best, 1 + f(new_d, dict, board, n)
                                                )
                                                if e4 > -1:
                                                    board[e4] -= 1
                                                board[s4] += 1
                                        if e3 > -1:
                                            board[e3] -= 1
                                        board[s3] += 1
                                if e2 > -1:
                                    board[e2] -= 1
                                board[s2] += 1
                        if e1 > -1:
                            board[e1] -= 1
                        board[s1] += 1
                expectation += 1 * best
            else:
                for s1 in range(0, n):
                    if board[s1] > 0:
                        board[s1] -= 1
                        e1 = s1 - d1
                        if e1 > -1:
                            board[e1] += 1
                        best = min(best, 1 + f(new_d, dict, board, n))
                        for s2 in range(0, n):
                            if board[s2] > 0:
                                board[s2] -= 1
                                e2 = s2 - d2
                                if e2 > -1:
                                    board[e2] += 1
                                best = min(best, 1 + f(new_d, dict, board, n))
                                if e2 > -1:
                                    board[e2] -= 1
                                board[s2] += 1
                        if e1 > -1:
                            board[e1] -= 1
                        board[s1] += 1
                expectation += best
    return expectation / 36


def expand(d, pips):
    new_d = {}
    board = [0 for _ in range(pips)]
    for i in range(0, pips):
        for k in d:
            for j in range(0, pips):
                board[j] = 0
            for index, checker_count in k:
                board[index] += checker_count
            board[i] += 1
            key = tuple((a, b) for a, b in enumerate(board) if b > 0)
            if key not in new_d:
                new_d[key] = f(new_d, d, board, n)
    return new_d | d


import datetime

if __name__ == "__main__":
    d = {}
    d[()] = 0
    checkers = 15
    n = 6
    for i in range(checkers):
        d = expand(d, n)
        now = datetime.datetime.now()
        print(now, i, len(d))
    print(len(d))
