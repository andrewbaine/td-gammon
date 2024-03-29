import argparse

import torch

import backgammon_env
import network
import policy
import tesauro
import try_gnubg


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=40)
    parser.add_argument("--model", required=True)
    parser.add_argument("--encoding", choices=["tesauro198"], required=True)
    parser.add_argument(
        "--softmax", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction)
    parser.add_argument("--games", type=int, default=10)

    args = parser.parse_args()
    bck = backgammon_env.Backgammon()
    observe = None

    match args.encoding:
        case "tesauro198":
            observe = tesauro.observe
        case _:
            assert False
    assert observe is not None

    t = observe(bck.s0(player_1=True))
    layers = [t.size()[0], args.hidden, 4]

    nn = network.layered(*layers, softmax=args.softmax)
    nn.load_state_dict(torch.load(args.model))
    nn = network.with_utility(nn)
    pol = policy.Policy_1_ply(bck, observe, nn)

    n = args.games

    results = try_gnubg.try_gnubg(pol, games=n, debug=args.debug)
    print(results)


if __name__ == "__main__":
    main()
