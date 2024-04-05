import torch

import argparse


def check_cuda(args):
    assert torch.cuda.is_available(), "cuda not available"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers(required=True)

    parser_cc = sp.add_parser("check-cuda", aliases=["cc"])
    parser_cc.set_defaults(func=check_cuda)
    args = parser.parse_args()
    args.func(args)
