import plyvel
from contextlib import nullcontext


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
    backgammon,
    done_check,
    eligibility_trace,
    move_vectors,
    td,
    training_config,
    encoders,
)


def init_parser(parser):
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--out", type=int)
    parser.add_argument(
        "--encoding",
        type=str,
        choices=["baine", "tesauro", "baine_epc"],
        default="baine_epc",
    )
    parser.add_argument("--force-cuda", action="store_true")
    parser.add_argument("--alpha", type=float, dest="α")
    parser.add_argument("--lambda", type=float, dest="λ")
    parser.add_argument("--continue", action="store_true", dest="cont")
    parser.add_argument("--fork", type=str)
    parser.add_argument("--epc-db", type=str)


def train(args):
    logging.basicConfig(encoding="utf-8", level=logging.INFO)
    device = torch.device(
        "cuda" if (args.force_cuda or torch.cuda.is_available()) else "cpu"
    )
    config_path = os.path.join(args.save_dir, "config.txt")
    start = 0
    starting_model = None
    if args.cont:
        config = training_config.load(config_path)
        fs = sorted(glob.glob("{dir}/model.*.pt".format(dir=args.save_dir)))
        if fs:
            m = re.match(r".*/model.(\d+).pt", fs[-1])
            if m:
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

    board: torch.Tensor = torch.tensor(
        [[0 for _ in range(27)]], dtype=torch.float, device=device
    )
    match config.out:
        case 4:
            utility = encoders.utility_tensor(device=device)
        case 6:
            utility = encoders.backgammon_utility_tensor(device=device)
        case _:
            assert False

    cm = nullcontext()
    match config.encoding:
        case "baine":
            encoder_builder = lambda ctx: encoders.Baine(device=device)
        case "tesauro":
            encoder_builder = lambda ctx: encoders.Tesauro(device=device)
        case "baine_epc":
            epc_db = args.epc_db
            assert epc_db is not None
            cm = plyvel.DB(epc_db, create_if_missing=False)
            places = [(1, 7), (7, 19), (19, 26)]

            encoder_builder = lambda ctx: encoders.BaineEPC(ctx, places, device=device)
        case _:
            assert False

    with cm:
        encoder = encoder_builder(cm)
        layers = [encoder(board).numel(), config.hidden, config.out]
        nn: torch.nn.Sequential = encoders.layered(*layers, device=device)
        if starting_model:
            nn.load_state_dict(torch.load(starting_model))

        move_checker = done_check.Donecheck(device=device)
        move_tensors = move_vectors.MoveTensors(device=device)

        et = eligibility_trace.ElibilityTrace(nn, α=config.α, λ=config.λ, device=device)
        evaluator = encoders.Evaluator(encoder, nn, utility)
        a = agent.OnePlyAgent(evaluator, move_tensors)

        temporal_difference = td.TD(
            move_checker,
            a,
            et,
            torch.tensor(
                [backgammon.make_board() + [0]], dtype=torch.float, device=device
            ),
        )

        for i in range(start, start + args.iterations):
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
