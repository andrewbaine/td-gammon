import struct
import plyvel


def make_key(board):
    end = len(board)
    while end > 0 and not board[end - 1]:
        end -= 1
    return bytes(board[1:end])


def inc(board):
    i = 1
    s = sum(board[1:])
    while True:
        if i == len(board):
            board.append(0)
        board[i] = board[i] + 1
        board[0] -= 1
        s += 1
        if s < 16:
            return
        s -= board[i]
        board[0] += board[i]
        board[i] = 0
        i += 1


def iterations(n, start):
    board = start
    while True:
        yield board
        inc(board)
        if len(board) > n + 1:
            return


def g(cache, db, board):
    key = make_key(board)
    if key in cache:
        return cache[key]
    v = db.get(key)
    assert v is not None
    (v,) = struct.unpack("f", v)
    return v


def f(cache, db, board):
    assert sum(board) == 15
    if board[0] == 15:
        return 0
    expectation = 0
    ec = 0
    board_pre = tuple(board)
    board_length = len(board)
    for d1 in range(1, 7):
        for d2 in range(d1, 7):
            factor = None
            best = None
            if d1 == d2:
                factor = 1
                for s1 in range(board_length - 1, 0, -1):
                    if board[s1] > 0:
                        board[s1] -= 1
                        e1 = max(s1 - d1, 0)
                        board[e1] += 1
                        y = g(cache, db, board)
                        best = y if best is None else min(best, y)
                        for s2 in range(s1, 0, -1):
                            if board[s2] > 0:
                                board[s2] -= 1
                                e2 = max(s2 - d1, 0)
                                board[e2] += 1
                                best = min(best, g(cache, db, board))
                                for s3 in range(s2, 0, -1):
                                    if board[s3] > 0:
                                        board[s3] -= 1
                                        e3 = max(s3 - d1, 0)
                                        board[e3] += 1
                                        best = min(best, g(cache, db, board))
                                        for s4 in range(s3, 0, -1):
                                            if board[s4] > 0:
                                                board[s4] -= 1
                                                e4 = max(s4 - d1, 0)
                                                board[e4] += 1
                                                best = min(best, g(cache, db, board))
                                                board[e4] -= 1
                                                board[s4] += 1
                                        board[e3] -= 1
                                        board[s3] += 1
                                board[e2] -= 1
                                board[s2] += 1
                        board[e1] -= 1
                        board[s1] += 1
            else:
                factor = 2
                for a, b in ((d1, d2), (d2, d1)):
                    for s1 in range(board_length - 1, 0, -1):
                        if board[s1] > 0:
                            e1 = max(s1 - a, 0)
                            board[s1] -= 1
                            board[e1] += 1
                            y = g(cache, db, board)
                            best = y if best is None else min(best, y)
                            for s2 in range(s1, 0, -1):
                                if board[s2] > 0:
                                    e2 = max(s2 - b, 0)
                                    board[s2] -= 1
                                    board[e2] += 1
                                    best = min(best, g(cache, db, board))
                                    board[e2] -= 1
                                    board[s2] += 1
                            board[e1] -= 1
                            board[s1] += 1
            assert best is not None
            assert factor == (1 if d1 == d2 else 2)
            expectation += factor * best
            ec += factor
            assert tuple(board) == board_pre
    assert ec == 36
    return 1 + expectation / ec


def main(n, db, start, batch_size):
    cache = {}
    with plyvel.DB(db, create_if_missing=True) as db:
        for board in iterations(n, start):
            y = f(cache, db, board)
            key = make_key(board)
            cache[key] = y
            print(board, "\t", list(key), "\t", y)
            if len(cache) >= batch_size:
                with db.write_batch() as wb:
                    for k, v in cache.items():
                        wb.put(k, struct.pack("f", v))
                cache.clear()
        with db.write_batch() as wb:
            for k, v in cache.items():
                wb.put(k, struct.pack("f", v))


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pips", type=int)
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--batch", type=int, default=100000)
    parser.add_argument("db", type=str)
    args = parser.parse_args()
    xs = [int(x) for x in args.start.split(",")] if args.start else []
    xs = [15]
    for x in args.start.split(",") if args.start else []:
        y = int(x)
        assert y >= 0
        xs.append(y)
        xs[0] -= y
    assert xs[0] > -1
    assert sum(xs) == 15
    assert xs[-1] > -1
    main(args.pips, args.db, xs, args.batch)
