import tesauro
import argparse
from os import listdir
from os.path import isfile, join
import re
import torch

import head_to_head
import network

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--dir", type=str, default="tesauro80")
    parser.add_argument("--hidden", type=int, default=80)
    args = parser.parse_args()

    p = "tesauro80"
    layers = [198, args.hidden, 4]

    network_1 = network.layered(*layers)

    network_2 = network.layered(*layers)

    observe = tesauro.observe

    files = [
        f
        for f in listdir(p)
        if isfile(join(p, f)) and re.match("model\\.\\d+0000.pt", f)
    ]

    d = {}
    for f1 in files:
        print(f1)
        if f1 not in d:
            d[f1] = [0, 0, 0, 0]
        f1_results = d[f1]
        network_1.load_state_dict(
            torch.load("{dir}/{file}".format(dir=args.dir, file=f1))
        )
        for f2 in files:
            if f1 < f2:
                if f2 not in d:
                    d[f2] = [0, 0, 0, 0]
                f2_results = d[f2]
                network_2.load_state_dict(
                    torch.load("{dir}/{file}".format(dir=args.dir, file=f2))
                )
                r = head_to_head.trial(
                    (network_1, observe),
                    (network_2, observe),
                    cb=lambda _: None,
                    games=args.games,
                )
                for i, x in enumerate(r):
                    f1_results[i] += x
                    f2_results[3 - i] += x

    results = [(-2 * v[0] - v[1] + v[2] + 2 * v[3], k, v) for k, v in d.items()]
    results.sort()
    for equity, k, v in results:
        print(k, equity, v)
