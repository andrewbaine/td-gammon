import struct
import plyvel

from td_gammon.epc import make_key
from td_gammon import backgammon_env


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


def f(bck, cache, db, board):
    # assert sum(board) == 15
    if board[0] == 15:
        return 0

    board = board + [0 for _ in range(26 - len(board))]
    # assert len(board) == 26

    expectation = 0
    ec = 0

    for d1 in range(1, 7):
        for d2 in range(d1, 7):
            state = (board, True, (d1, d2))
            best = None
            for m in bck.available_moves(state):
                (b2, _, _) = bck.next(state, m)
                v = g(cache, db, b2)
                if best is None or v < best:
                    best = v
            # assert best is not None
            factor = 1 if d1 == d2 else 2
            expectation += best * factor
            ec += factor

    # assert ec == 36
    return 1 + expectation / ec


import random


def main(n, db, start, batch_size):
    cache = {}
    bck = backgammon_env.Backgammon()
    with plyvel.DB(db, create_if_missing=True) as db:
        for board in iterations(n, start):
            y = f(bck, cache, db, board)
            key = make_key(board)
            cache[key] = y
            if random.random() < 0.0005:
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
