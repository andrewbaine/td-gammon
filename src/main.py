import argparse
import sys

import torch

import player
import train
import write_move_tensors


def play(args):
    player.play(sys.stdin)


def check_cuda(args):
    assert torch.cuda.is_available(), "cuda not available"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers(required=True)

    parser_cc = sp.add_parser("check-cuda", aliases=["cc"])
    parser_cc.set_defaults(func=check_cuda)

    parser_train = sp.add_parser("train")
    train.init_parser(parser_train)
    parser_train.set_defaults(func=train.train)

    parser_play = sp.add_parser("play")
    parser_play.set_defaults(func=play)

    parser_move_tensors = sp.add_parser("move-tensors")
    write_move_tensors.init_parser(parser_move_tensors)
    parser_move_tensors.set_defaults(func=write_move_tensors.main)

    args = parser.parse_args()
    if torch.cuda.is_available():
        with torch.cuda.device("cuda"):
            args.func(args)
    else:
        args.func(args)
