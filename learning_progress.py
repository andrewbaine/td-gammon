import argparse
from os import listdir
from os.path import isfile, join

import torch

import backgammon_env
import network
import policy
import tesauro
from try_gnubg import try_gnubg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--hidden", type=int, default=80)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument(
        "--softmax", action=argparse.BooleanOptionalAction, default=False
    )

    args = parser.parse_args()

    layers = [198, args.hidden, 4]

    nn = network.layered(*layers, softmax=args.softmax)

    observe = tesauro.observe

    bck = backgammon_env.Backgammon()

    files = [
        f
        for f in listdir(args.dir)
        if isfile(join(args.dir, f))
        and int(f.split(".")[1]) >= args.start
        and (int(f.split(".")[1]) % args.step) == 0
    ]

    for f in files:
        nn.load_state_dict(torch.load("{dir}/{file}".format(dir=args.dir, file=f)))
        pol = policy.Policy_1_ply(bck, observe, network.with_utility(nn))
        results = try_gnubg(pol, games=args.games, debug=False)
        assert len(results) == 6
        rate = sum([i[0] * i[1] for i in zip(results, [-3, -2, -1, 1, 2, 3])]) / sum(
            results
        )
        key = int(f.split(".")[1])
        value = rate
        print("{key}\t{value}".format(key=key, value=value))
