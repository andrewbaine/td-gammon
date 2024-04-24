import argparse
import os.path
from pathlib import Path

import torch

from td_gammon import agent, backgammon, player, move_vectors, training_config, encoders


def main(args):
    device = torch.device(
        "cuda" if (args.force_cuda or torch.cuda.is_available()) else "cpu"
    )

    model_path = args.load_model
    config_path = os.path.join(Path(model_path).parent.absolute(), "config.txt")
    config = training_config.load(config_path)

    match config.out:
        case 4:
            utility = encoders.utility_tensor(device=device)
        case 6:
            utility = encoders.backgammon_utility_tensor(device=device)
        case _:
            assert False

    match config.encoding:
        case "baine":
            encoder = encoders.Baine()
        case "tesauro":
            encoder = encoders.Tesauro()
        case _:
            assert False

    board = torch.tensor(
        [backgammon.make_board() + [1]], dtype=torch.float, device=device
    )
    t = encoder(board)
    (m, n) = t.size()
    assert m == 1

    layers = [n, config.hidden, config.out]
    nn: torch.nn.Sequential = encoders.layered(*layers, device=device)
    nn.load_state_dict(torch.load(model_path))
    move_tensors = move_vectors.MoveTensors(device=device)

    device = torch.device("cpu")

    evaluator = encoders.Evaluator(encoder, nn, utility)

    a = agent.OnePlyAgent(evaluator, move_tensors)
    player.play(a, args.games, device)


def init_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--load-model", type=str, required=True)
    parser.add_argument("--force-cuda", action="store_true")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init_parser(parser)
    args = parser.parse_args()
    main(args)
