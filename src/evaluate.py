import argparse
import os.path
from pathlib import Path

import torch

import agent
import backgammon
import network
import player
import read_move_tensors
import training_config

import encoders


def main(args):

    model_path = args.load_model
    config_path = os.path.join(Path(model_path).parent.absolute(), "config.txt")
    config = training_config.load(config_path)

    match config.out:
        case 4:
            utility = network.utility_tensor()
        case 6:
            utility = network.backgammon_utility_tensor()
        case _:
            assert False

    match config.encoding:
        case "baine":
            encoder = encoders.Baine()
        case "tesauro":
            encoder = encoders.Tesauro()
        case _:
            assert False

    board = torch.tensor(backgammon.make_board() + [1], dtype=torch.float)
    t = encoder(board)
    (m, n) = t.size()
    assert m == 1

    layers = [n, config.hidden, config.out]
    nn: torch.nn.Sequential = network.layered(*layers)
    nn.load_state_dict(torch.load(model_path))
    move_tensors = read_move_tensors.MoveTensors()

    device = torch.device("cpu")
    if torch.cuda.is_available() or args.force_cuda:
        device = torch.device("cuda")
        nn = nn.to(device)
        move_tensors.to_(device)
        encoder.to_(device)
        utility = utility.to(device)

    evaluator = encoders.Evaluator(encoder, nn, utility)

    a = agent.OnePlyAgent(evaluator, move_tensors)
    player.play(a, args.games)


def init_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--load-model", type=str, required=True)
    parser.add_argument("--force-cuda", type=bool, default=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init_parser(parser)
    args = parser.parse_args()
    main(args)
