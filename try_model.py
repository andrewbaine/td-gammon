import argparse

import torch

import backgammon_env
import network
import tesauro

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("m1")
    parser.add_argument("m2")
    parser.add_argument("--games", type=int, default=1)
    args = parser.parse_args()

    layers = [198, 40, 4]

    good = network.layered(*layers)
    good.load_state_dict(torch.load(args.m1))

    bad = network.layered(*layers)
    bad.load_state_dict(torch.load(args.m2))

    bck = backgammon_env.Backgammon()
    observer = tesauro.Tesauro198()

    import head_to_head

    results = head_to_head.trial(
        (good, observer),
        (bad, observer),
        cb=lambda results: print(
            results,
            (-2 * results[0] + -1 * results[1] + results[2] + 2 * results[3])
            / (sum(results)),
        ),
        games=args.games,
    )
