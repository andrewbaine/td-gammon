import argparse
import backgammon
import done_check
import network

import read_move_tensors
import td
import tesauro

import torch
import logging


def init_parser(parser):
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=40)
    parser.add_argument("--move-tensors", type=str, default="move_tensors/current")


def train(args):
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename="example.log", encoding="utf-8", level=logging.INFO)
    layers = [198, args.hidden, 6]
    nn = network.layered(*layers, softmax=True)
    board = torch.tensor(backgammon.make_board(), dtype=torch.float)
    move_checker = done_check.Donecheck()
    move_tensors = read_move_tensors.MoveTensors(args.move_tensors)
    encoder = tesauro.Encoder()
    temporal_difference = td.TD(board, move_checker, move_tensors, nn, encoder)

    for i in range(args.iterations):
        if i % 1000 == 0:
            filename = "{%07d}".format()
        temporal_difference.episode()


logger.debug("This message should go to the log file")
logger.info("So should this")
logger.warning("And this, too")
logger.error("And non-ASCII stuff, too, like Øresund and Malmö")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init_parser(parser)
    args = parser.parse_args()
    if torch.cuda.is_available():
        with torch.device("cuda"):
            train(args)
    else:
        train(args)
