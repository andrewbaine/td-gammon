import argparse
import glob
import logging
import os
import os.path
import re
import shutil

import torch

from td_gammon import (
    agent,
    done_check,
    eligibility_trace,
    move_vectors,
    td,
    training_config,
    encoders,
)


def init_parser(parser):
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--out", type=int)
    parser.add_argument(
        "--encoding", type=str, choices=["baine", "tesauro"], default="baine"
    )
    parser.add_argument("--force-cuda", type=bool, default=False)
    parser.add_argument("--alpha", type=float, dest="α")
    parser.add_argument("--lambda", type=float, dest="λ")
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
        starting_model = os.path.join(args.save_dir, filename)
        shutil.copyfile(args.fork, starting_model)
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

    board: torch.Tensor = torch.tensor([[0 for _ in range(27)]], dtype=torch.float)
    match config.out:
        case 4:
            utility = encoders.utility_tensor()
        case 6:
            utility = encoders.backgammon_utility_tensor()
        case _:
            assert False

    match config.encoding:
        case "baine":
            encoder = encoders.Baine()
        case "tesauro":
            encoder = encoders.Tesauro()
        case _:
            assert False
    layers = [encoder(board).numel(), config.hidden, config.out]
    nn: torch.nn.Sequential = encoders.layered(*layers)
    if start:
        assert starting_model
        nn.load_state_dict(torch.load(starting_model))

    move_checker = done_check.Donecheck()
    move_tensors = move_vectors.MoveTensors()

    et = eligibility_trace.ElibilityTrace(nn, α=config.α, λ=config.λ)
    evaluator = encoders.Evaluator(encoder, nn, utility)
    a = agent.OnePlyAgent(evaluator, move_tensors)
    temporal_difference = td.TD(move_checker, a, et)

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
