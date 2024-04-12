import baine_encoding
import agent
import argparse
import backgammon
import done_check
import network

import read_move_tensors
import td
import tesauro

import torch
import logging

import os

import eligibility_trace
import contextlib


def init_parser(parser):
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=40)
    parser.add_argument("--move-tensors", type=str, default="var/move_tensors/current")
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--out", type=int, required=True)
    parser.add_argument(
        "--encoding", type=str, required=True, choices=["baine", "tesauro"]
    )


def train(args):
    logging.basicConfig(encoding="utf-8", level=logging.INFO)
    cm = (
        torch.cuda.device(torch.device("cuda"))
        if torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    with cm:
        board: torch.Tensor = torch.tensor(backgammon.make_board(), dtype=torch.float)
        match args.encoding:
            case "baine":
                encoder = baine_encoding.Encoder()
            case "tesauro":
                encoder = tesauro.Encoder()
            case _:
                assert False
        t = encoder.encode(board, True)
        layers = [t.numel(), args.hidden, args.out]
        nn: torch.nn.Sequential = network.layered(*layers)
        move_checker = done_check.Donecheck()
        move_tensors = read_move_tensors.MoveTensors(args.move_tensors)

    a = agent.OnePlyAgent(nn, move_tensors, encoder, out=args.out)

    et = eligibility_trace.ElibilityTrace(nn)
    temporal_difference = td.TD(board, move_checker, a, et)

    os.makedirs(args.save_dir, exist_ok=True)
    for i in range(args.iterations):
        if i % 100 == 0:
            print(i)
        if i % 1000 == 0:
            filename = "{dir}/model.{i:08d}.pt".format(i=i, dir=args.save_dir)
            torch.save(nn.state_dict(), f=filename)
        temporal_difference.episode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init_parser(parser)
    args = parser.parse_args()
    train(args)
