import argparse
import sys


import player
import train
import torch


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

    args = parser.parse_args()
    args.func(args)
