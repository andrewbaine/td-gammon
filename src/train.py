import argparse
import backgammon
import done_check
import network

import read_move_tensors
import td
import tesauro

import torch
import logging

import os


def init_parser(parser):
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=40)
    parser.add_argument("--move-tensors", type=str, default="move_tensors/current")
    parser.add_argument("--save-dir", type=str, required=True)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(filename="example.log", encoding="utf-8", level=logging.INFO)
    layers = [198, args.hidden, 6]
    nn: torch.nn.Sequential = network.layered(*layers, softmax=True, device=device)
    nn.to(device)

    board: torch.Tensor = torch.tensor(backgammon.make_board(), dtype=torch.float)
    board.to(device)

    move_checker = done_check.Donecheck(device=device)
    move_tensors = read_move_tensors.MoveTensors(args.move_tensors, device=device)
    encoder = tesauro.Encoder(device=device)
    temporal_difference = td.TD(
        board, move_checker, move_tensors, nn, encoder, device=device
    )

    os.makedirs(args.save_dir, exist_ok=True)
    for i in range(args.iterations):
        if i % 1000 == 0:
            filename = "{dir}/model.{i:08d}.pt".format(i=i, dir=args.save_dir)
            torch.save(nn.state_dict(), f=filename)
        temporal_difference.episode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init_parser(parser)
    args = parser.parse_args()
    train(args)
