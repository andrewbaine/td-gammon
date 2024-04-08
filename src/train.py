import argparse
import backgammon
import done_check
import network

import read_move_tensors
import td
import tesauro

import torch


def init_parser(parser):
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=40)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--move-tensors", type=str, required=True)


def train(args):
    device = args.device
    layers = [198, args.hidden, 4]
    nn = network.layered(*layers, softmax=True)
    board = torch.tensor(backgammon.make_board(), dtype=torch.float, device=device)
    move_checker = done_check.Donecheck(device)
    move_tensors = read_move_tensors.MoveTensors(args.move_tensors, device)
    encoder = tesauro.Encoder(device=device)
    temporal_difference = td.TD(board, move_checker, move_tensors, nn, encoder)
    temporal_difference.episode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init_parser(parser)
    args = parser.parse_args
    train(args)
