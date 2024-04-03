import argparse
from datetime import datetime
import os
import os.path
import bt_2
import torch


def singles_dir(prefix, d1):
    return os.path.join(prefix, "singles", str(d1))


def doubles_dir(prefix, d1, name):
    return os.path.join(prefix, "doubles", str(d1), name)


def ab_dir(prefix, d1, d2):
    return os.path.join(prefix, "ab", str(d1) + str(d2))


def write(path, x):
    for tensor, name in zip(
        bt_2.tensorize(x), ("moves.pt", "low.pt", "high.pt", "vector.pt")
    ):
        torch.save(tensor, os.path.join(path, name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix")
    args = parser.parse_args()
    prefix = args.prefix
    if not prefix:
        current_date = datetime.now()
        prefix = "move_tensors/{date}".format(date=current_date.isoformat())

    os.makedirs(prefix, exist_ok=False)
    for d1 in range(1, 7):
        dir = singles_dir(prefix, d1)
        os.makedirs(dir)
        write(dir, bt_2.all_moves_die(d1))
        for t, name in zip(bt_2.all_doubles(d1), ("1", "2", "3", "4")):
            p = doubles_dir(prefix, d1, name)
            os.makedirs(p)
            write(p, t)
        for d2 in range(1, d1):
            p = ab_dir(prefix, d1, d2)
            os.makedirs(p)
            write(p, bt_2.all_moves_dice(d1, d2))
