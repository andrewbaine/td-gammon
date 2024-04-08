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
    parser.add_argument("--move-tensors", type=str, required=True)


def train(args):
    layers = [198, args.hidden, 4]
    nn = network.layered(*layers, softmax=True)
    board = torch.tensor(backgammon.make_board(), dtype=torch.float)
    move_checker = done_check.Donecheck()
    move_tensors = read_move_tensors.MoveTensors(args.move_tensors)
    encoder = tesauro.Encoder()
    temporal_difference = td.TD(board, move_checker, move_tensors, nn, encoder)

    for i in range(args.iterations):
        temporal_difference.episode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init_parser(parser)
    args = parser.parse_args()
    if torch.cuda.is_available():
        with torch.device("cuda"):
            train(args)
    else:
        train(args)
