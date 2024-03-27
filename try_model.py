import argparse

import torch

import backgammon_env
import head_to_head
import network
import tesauro

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-1", type=int, default=40)
    parser.add_argument("--hidden-2", type=int, default=40)

    parser.add_argument(
        "--softmax-1", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--softmax-2", action=argparse.BooleanOptionalAction, default=False
    )

    parser.add_argument("m1")
    parser.add_argument("m2")
    parser.add_argument("--games", type=int, default=1)
    args = parser.parse_args()

    l1 = [198, args.hidden_1, 4]
    l2 = [198, args.hidden_2, 4]

    network_1 = network.layered(*l1, softmax=args.softmax_1)
    network_1.load_state_dict(torch.load(args.m1))

    network_2 = network.layered(*l2, softmax=args.softmax_2)
    network_2.load_state_dict(torch.load(args.m2))

    bck = backgammon_env.Backgammon()
    observe = tesauro.observe

    results = head_to_head.trial(
        (network_1, observe),
        (network_2, observe),
        cb=lambda results: print(
            results,
            (-2 * results[0] + -1 * results[1] + results[2] + 2 * results[3])
            / (sum(results)),
        ),
        games=args.games,
    )
    print(
        (-2 * results[0] + -1 * results[1] + results[2] + 2 * results[3])
        / (sum(results)),
        args.m1,
        "v",
        args.m2,
    )
