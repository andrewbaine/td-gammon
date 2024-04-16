import argparse
import glob
import logging
import os
import os.path
import re
import shutil

import torch

import agent
import backgammon
import baine_encoding
import done_check
import eligibility_trace
import network
import read_move_tensors
import td
import tesauro
import training_config


def init_parser(parser):
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=40)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--out", type=int, default=4)
    parser.add_argument(
        "--encoding", type=str, choices=["baine", "tesauro"], default="baine"
    )
    parser.add_argument("--force-cuda", type=bool, default=False)
    parser.add_argument("--alpha", type=float, default=0.1, dest="α")
    parser.add_argument("--lambda", type=float, default=0.7, dest="λ")
    parser.add_argument("--continue", action="store_true", dest="cont")
    parser.add_argument("--fork", type=str)


def train(args):
    logging.basicConfig(encoding="utf-8", level=logging.INFO)
    config_path = os.path.join(args.save_dir, "config.txt")
    start = 0
    starting_model = None
    if args.cont:
        config = training_config.load(config_path)
        fs = sorted(glob.glob("{dir}/model.*.pt".format(dir=args.save_dir)))
        if fs:
            m = re.match(r".*/model.(\d+).pt", fs[-1])
            assert m
            start = int(m.group(1))
            starting_model = m.group(0)
            logging.info(
                "continuing from config {x} model {y}".format(
                    x=config_path, y=starting_model
                )
            )
    elif args.fork:

        filename = os.path.basename(args.fork)
        os.makedirs(args.save_dir, exist_ok=False)
        shutil.copyfile(args.fork, os.path.join(args.save_dir, filename))
        m = re.match(r"model.(\d+).pt", filename)
        assert m
        start = int(m.group(1))
        config = training_config.from_parent(
            training_config.load(
                os.path.join(os.path.dirname(args.fork), "config.txt")
            ),
            args,
        )
        training_config.store(config, config_path)
    else:
        config = training_config.from_args(args)
        os.makedirs(args.save_dir, exist_ok=False)
        training_config.store(config, config_path)

    board: torch.Tensor = torch.tensor(backgammon.make_board(), dtype=torch.float)
    match config.out:
        case 4:
            utility = network.utility_tensor()
        case 6:
            utility = network.backgammon_utility_tensor()
        case _:
            assert False

    match config.encoding:
        case "baine":
            encoder = baine_encoding.Encoder()
        case "tesauro":
            encoder = tesauro.Encoder()
        case _:
            assert False
    layers = [encoder.encode(board, True).numel(), config.hidden, config.out]
    nn: torch.nn.Sequential = network.layered(*layers)
    if start:
        assert starting_model
        nn.load_state_dict(torch.load(starting_model))

    move_checker = done_check.Donecheck()

    move_tensors = read_move_tensors.MoveTensors()

    if torch.cuda.is_available() or args.force_cuda:
        device = torch.device("cuda")
        board = board.to(device=device)
        nn = nn.to(device=device)
        move_checker.to_(device=device)
        move_tensors.to_(device=device)
        encoder.to_(device=device)
        utility = utility.to(device=device)

    et = eligibility_trace.ElibilityTrace(nn, α=config.α, λ=config.λ)
    a = agent.OnePlyAgent(nn, move_tensors, encoder, utility=utility)
    temporal_difference = td.TD(board, move_checker, a, et)

    for i in range(start, config.iterations):
        if i % 1000 == 0:
            filename = "{dir}/model.{i:08d}.pt".format(i=i, dir=args.save_dir)
            torch.save(nn.state_dict(), f=filename)
        temporal_difference.episode()
    filename = "{dir}/model.{i:08d}.pt".format(i=i, dir=args.save_dir)
    torch.save(nn.state_dict(), f=filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init_parser(parser)
    args = parser.parse_args()
    train(args)
