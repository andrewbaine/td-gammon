import argparse
from contextlib import nullcontext
import os.path
from pathlib import Path

import plyvel
import torch

from td_gammon import agent, backgammon, encoders, move_vectors, player, training_config


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

    cm = nullcontext()
    match config.encoding:
        case "baine":
            encoder_builder = lambda ctx: encoders.Baine(device=device)
        case "tesauro":
            encoder_builder = lambda ctx: encoders.Tesauro(device=device)
        case "baine_epc":
            epc_db = args.epc_db
            assert epc_db is not None
            cm = plyvel.DB(epc_db, create_if_missing=False)
            places = [(1, 7), (7, 19), (19, 26)]

            encoder_builder = lambda ctx: encoders.BaineEPC(ctx, places, device=device)
        case _:
            assert False

    board = torch.tensor(
        [backgammon.make_board() + [1]], dtype=torch.float, device=device
    )
    with cm:
        encoder = encoder_builder(cm)
        t = encoder(board)
        (m, n) = t.size()
        assert m == 1

        layers = [n, config.hidden, config.out]
        nn: torch.nn.Sequential = encoders.layered(*layers, device=device)
        nn.load_state_dict(torch.load(model_path, map_location=device))
        move_tensors = move_vectors.MoveTensors(device=device)

        evaluator = encoders.Evaluator(encoder, nn, utility)

        a = agent.OnePlyAgent(evaluator, move_tensors)
        player.play(a, args.games, device)


def init_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--load-model", type=str, required=True)
    parser.add_argument("--force-cuda", action="store_true")
    parser.add_argument("--epc-db", type=str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init_parser(parser)
    args = parser.parse_args()
    main(args)
