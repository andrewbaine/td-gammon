import struct

import plyvel


def lookup(db_path, board):
    with plyvel.DB(db_path, create_if_missing=False) as db:
        k = bytes(board)
        v = db.get(k)
        (v,) = struct.unpack("f", v)
        print(v, v * 49 / 6)


def count(db_path):
    with plyvel.DB(db_path, create_if_missing=False) as db:
        with db.iterator() as it:
            count = 0
            for _ in it:
                count += 1
            print(count)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, required=True)
    parser.add_argument("--board", default="")
    parser.add_argument("action", choices=["count", "lookup"])

    args = parser.parse_args()
    match args.action:
        case "count":
            count(args.db)
        case "lookup":
            board = [int(x) for x in args.board.split(",")] if args.board else []
            lookup(args.db, board)
