import argparse

import torch

import evaluate
import train
import write_move_tensors


def check_cuda(_):
    assert torch.cuda.is_available(), "cuda not available"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers(required=True)

    parser_cc = sp.add_parser("check-cuda", aliases=["cc"])
    parser_cc.set_defaults(func=check_cuda)

    parser_train = sp.add_parser("train")
    train.init_parser(parser_train)
    parser_train.set_defaults(func=train.train)

    parser_evaluate = sp.add_parser("evaluate")
    evaluate.init_parser(parser_evaluate)
    parser_evaluate.set_defaults(func=evaluate.main)

    parser_move_tensors = sp.add_parser("move-tensors")
    write_move_tensors.init_parser(parser_move_tensors)
    parser_move_tensors.set_defaults(func=write_move_tensors.main)

    args = parser.parse_args()
    args.func(args)
