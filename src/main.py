import td
import done_check
import torch
import tesauro
import argparse
import backgammon
import network
import read_move_tensors


def check_cuda(args):
    assert torch.cuda.is_available(), "cuda not available"


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
    sp = parser.add_subparsers(required=True)

    parser_cc = sp.add_parser("check-cuda", aliases=["cc"])
    parser_cc.set_defaults(func=check_cuda)

    parser_train = sp.add_parser("train")
    parser_train.set_defaults(func=train)
    parser_train.add_argument("--iterations", type=int, default=100)
    parser_train.add_argument("--hidden", type=int, default=40)
    parser_train.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser_train.add_argument("--move-tensors", type=str, required=True)

    args = parser.parse_args()
    args.func(args)
