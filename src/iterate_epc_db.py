import struct

import plyvel


def main(db_path, board):

    with plyvel.DB(db_path, create_if_missing=True) as db:
        k = bytes(board)
        v = db.get(k)
        (v,) = struct.unpack("f", v)
        print(v, v * 49 / 6)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str)
    parser.add_argument("board", default="")
    args = parser.parse_args()
    board = [int(x) for x in args.board.split(",")] if args.board else []
    main(args.db, board)
