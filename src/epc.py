import struct
import plyvel

from td_gammon.epc import make_key


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
    (v,) = struct.unpack("f", v)
    return v


def f(cache, db, board):
    if board[0] == 15:
        return 0
    greatest_pip = len(board) - 1
    #    board_before = tuple(board)
    count = 0
    sum = 0
    for d1 in range(1, 7):
        for d2 in range(d1, 7):
            gp = greatest_pip
            best = None
            if d1 == d2:
                factor = 1
                for s1 in range(len(board) - 1, 0, -1):
                    if board[s1] == 0:
                        if gp == s1:
                            gp -= 1
                    else:
                        #                        assert board[s1] > 0
                        e1 = s1 - d1
                        if e1 > 0 or (e1 == 0 and gp < 7) or gp == s1:
                            e1 = max(e1, 0)
                            board[s1] -= 1
                            board[e1] += 1
                            v = g(cache, db, board)
                            if best is None or v < best:
                                best = v
                            for s2 in range(s1, 0, -1):
                                if board[s2] == 0:
                                    if gp == s2:
                                        gp -= 1
                                else:
                                    e2 = s2 - d1
                                    if e2 > 0 or (e2 == 0 and gp < 7) or gp == s2:
                                        e2 = max(e2, 0)
                                        board[s2] -= 1
                                        board[e2] += 1
                                        v = g(cache, db, board)
                                        #                                        assert best is not None
                                        if v < best:
                                            best = v
                                        for s3 in range(s2, 0, -1):
                                            if board[s3] == 0:
                                                if gp == s3:
                                                    gp -= 1
                                            else:
                                                #                                                assert board[s3] > 0
                                                e3 = s3 - d1
                                                if (
                                                    e3 > 0
                                                    or (e3 == 0 and gp < 7)
                                                    or (e3 < 0 and gp == s3)
                                                ):
                                                    e3 = max(e3, 0)
                                                    board[s3] -= 1
                                                    board[e3] += 1
                                                    v = g(cache, db, board)
                                                    #                                                    assert best is not None
                                                    if v < best:
                                                        best = v
                                                    for s4 in range(s3, 0, -1):
                                                        if board[s4] == 0:
                                                            if gp == s4:
                                                                gp -= 1
                                                        else:
                                                            #                                                            assert board[s4] > 0
                                                            e4 = s4 - d1
                                                            if (
                                                                e4 > 0
                                                                or (e4 == 0 and gp < 7)
                                                                or gp == s4
                                                            ):
                                                                e4 = max(e4, 0)
                                                                board[s4] -= 1
                                                                board[e4] += 1
                                                                v = g(
                                                                    cache,
                                                                    db,
                                                                    board,
                                                                )
                                                                #                                                                assert best is not None
                                                                if v < best:
                                                                    best = v
                                                                board[e4] -= 1
                                                                board[s4] += 1
                                                                gp = max(gp, s4)
                                                    board[e3] -= 1
                                                    board[s3] += 1
                                                    gp = max(gp, s3)
                                        board[e2] -= 1
                                        board[s2] += 1
                                        gp = max(gp, s2)
                            board[e1] -= 1
                            board[s1] += 1
                            gp = max(gp, s1)
            #                    assert tuple(board) == board_before
            else:
                factor = 2
                for a, b in [(d1, d2), (d2, d1)]:
                    gp = greatest_pip
                    for s1 in range(len(board) - 1, 0, -1):
                        if board[s1] == 0:
                            if gp == s1:
                                gp -= 1
                        else:
                            #                            assert board[s1] > 0
                            e1 = s1 - a
                            if e1 > 0 or (e1 == 0 and gp < 7) or (e1 < 0 and gp == s1):
                                e1 = max(e1, 0)
                                board[s1] -= 1
                                board[e1] += 1
                                v = g(cache, db, board)
                                if best is None or v < best:
                                    best = v
                                for s2 in range(s1, 0, -1):
                                    if board[s2] == 0:
                                        if gp == s2:
                                            gp -= 1
                                    else:
                                        #                                        assert board[s2] > 0
                                        e2 = s2 - b
                                        if (
                                            e2 > 0
                                            or (e2 == 0 and gp < 7)
                                            or (e2 < 0 and gp == s2)
                                        ):
                                            e2 = max(e2, 0)
                                            board[s2] -= 1
                                            board[e2] += 1
                                            v = g(cache, db, board)
                                            #                                            assert v is not None
                                            if v < best:
                                                best = v
                                            board[e2] -= 1
                                            board[s2] += 1
                                            gp = max(gp, s2)
                                board[e1] -= 1
                                board[s1] += 1
                                gp = max(gp, s1)
            #                        assert tuple(board) == board_before
            #            assert best is not None
            sum += factor * best
            count += factor
    return 1 + sum / count


import random


def main(n, db, start, batch_size):
    cache = {}
    with plyvel.DB(db, create_if_missing=True) as db:
        for board in iterations(n, start):
            y = f(cache, db, board)
            key = make_key(board)
            cache[key] = y
            if random.random() < 0.001:
                print(board, "\t", list(key), "\t", y)
            if len(cache) >= batch_size:
                with db.write_batch() as wb:
                    for k, v in cache.items():
                        wb.put(k, struct.pack("f", v))
        with db.write_batch() as wb:
            for k, v in cache.items():
                wb.put(k, struct.pack("f", v))


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pips", type=int)
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--batch", type=int, default=10000000)
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
