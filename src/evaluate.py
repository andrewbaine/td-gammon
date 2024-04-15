import argparse
import os.path
from pathlib import Path

import torch

import agent
import agent
import backgammon
import baine_encoding
import network
import player
import read_move_tensors
import tesauro
import training_config


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
            encoder = baine_encoding.Encoder()
        case "tesauro":
            encoder = tesauro.Encoder()
        case _:
            assert False

    t = encoder.encode(torch.tensor(backgammon.make_board(), dtype=torch.float), True)

    layers = [t.numel(), config.hidden, config.out]
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

    a = agent.OnePlyAgent(nn, move_tensors, encoder, utility=utility)
    player.play(a, args.games, device=device)


def init_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--load-model", type=str, required=True)
    parser.add_argument("--force-cuda", type=bool, default=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init_parser(parser)
    args = parser.parse_args()
    main(args)
