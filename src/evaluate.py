import agent
import argparse
import player

import torch
import network
import agent
import read_move_tensors
import tesauro


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers = [198, args.hidden, args.out]
    nn: torch.nn.Sequential = network.layered(*layers, softmax=True)
    encoder = tesauro.Encoder(device=device)
    move_tensors = read_move_tensors.MoveTensors(args.move_tensors, device=device)
    a = agent.OnePlyAgent(nn, move_tensors, encoder, device=device, out=args.out)
    player.play(a, args.games)


def init_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=40)
    parser.add_argument("--move-tensors", type=str, default="move_tensors/current")
    parser.add_argument("--load-model", type=str, required=True)
    parser.add_argument("--out", type=int, default=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init_parser(parser)
    args = parser.parse_args()
    main(args)
