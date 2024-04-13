import backgammon
import agent
import argparse
import player

import torch
import network
import agent
import read_move_tensors
import tesauro
import baine_encoding

import os.path
from pathlib import Path


def main(args):

    model_path = args.load_model
    config_path = os.path.join(Path(model_path).parent.absolute(), "config.txt")
    with open(config_path, "r") as input:
        for line in input:
            tokens = [x.strip() for x in line.split("=")]
            assert len(tokens) == 2
            [key, value] = tokens
            match key:
                case "encoding":
                    encoding = value
                case "hidden":
                    hidden = int(value)
                case "out":
                    out = int(value)
                case "move-tensors":
                    move_tensors = value
                case "alpha" | "lambda" | "iterations":
                    pass
                case _:
                    raise Exception("unknown key " + key)

    match out:
        case 4:
            utility = network.utility_tensor()
        case 6:
            utility = network.backgammon_utility_tensor()
        case _:
            assert False

    match encoding:
        case "baine":
            encoder = baine_encoding.Encoder()
        case "tesauro":
            encoder = tesauro.Encoder()
        case _:
            assert False

    t = encoder.encode(torch.tensor(backgammon.make_board(), dtype=torch.float), True)

    layers = [t.numel(), hidden, out]
    nn: torch.nn.Sequential = network.layered(*layers)
    nn.load_state_dict(torch.load(model_path))
    move_tensors = read_move_tensors.MoveTensors(move_tensors)

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
    parser.add_argument("--move-tensors", type=str, default="/var/move_tensors/current")
    parser.add_argument("--load-model", type=str, required=True)
    parser.add_argument("--force-cuda", type=bool, default=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init_parser(parser)
    args = parser.parse_args()
    main(args)
