import struct

import plyvel


def f(v):
    (v,) = struct.unpack("f", v)
    return v


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("db1", type=str)
    parser.add_argument("db2", type=str)
    args = parser.parse_args()

    with plyvel.DB(args.db1, create_if_missing=False) as db1:
        with plyvel.DB(args.db2, create_if_missing=False) as db2:
            with db1.iterator() as it:
                for k, v in it:
                    v2 = db2.get(k)
                    if f(v) != f(v2):
                        print(list(k), f(v), f(v2))
