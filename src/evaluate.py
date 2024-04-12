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


def main(args):

    match args.out:
        case 4:
            utility = network.utility_tensor()
        case 6:
            utility = network.backgammon_utility_tensor()
        case _:
            assert False

    match args.encoding:
        case "baine":
            encoder = baine_encoding.Encoder()
        case "tesauro":
            encoder = tesauro.Encoder()
        case _:
            assert False

    t = encoder.encode(torch.tensor(backgammon.make_board(), dtype=torch.float), True)

    layers = [t.numel(), args.hidden, args.out]
    nn: torch.nn.Sequential = network.layered(*layers)
    nn.load_state_dict(torch.load(args.load_model))
    move_tensors = read_move_tensors.MoveTensors(args.move_tensors)

    if torch.cuda.is_available() or args.force_cuda:
        device = torch.device("cuda")
        nn = nn.to(device)
        move_tensors.to_(device)
        encoder.to_(device)
        utility = utility.to(device)

    a = agent.OnePlyAgent(nn, move_tensors, encoder, utility=utility)
    player.play(a, args.games)


def init_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=40)
    parser.add_argument("--move-tensors", type=str, default="move_tensors/current")
    parser.add_argument("--load-model", type=str, required=True)
    parser.add_argument("--out", type=int, default=4)
    parser.add_argument("--force-cuda", type=bool, default=False)
    parser.add_argument(
        "--encoding", type=str, required=True, choices=["baine", "tesauro"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init_parser(parser)
    args = parser.parse_args()
    main(args)
