import slow_but_right
import backgammon_env
import network
from model import Trainer
import torch
import argparse

from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=300000)
    parser.add_argument("--encoding", choices=["tesauro198"], required=True)
    parser.add_argument("--hidden", type=int, default=40)
    parser.add_argument(
        "--softmax", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("name")
    args = parser.parse_args()

    Path(args.name).mkdir(parents=True, exist_ok=True)

    observer = None
    tensor = None
    match args.encoding:
        case "tesauro198":
            observe = slow_but_right.tesauro_encode
        case _:
            assert False
    assert observe is not None

    args = parser.parse_args()
    bck = backgammon_env.Backgammon()
    t = observe(bck.s0())
    layers = [len(t), args.hidden, 4]
    nn = network.layered(*layers, softmax=args.softmax, device=torch.device("cpu"))
    trainer = Trainer(bck, nn, observe)

    results = [0, 0, 0, 0, 0, 0]
    for i in range(0, args.iterations):
        if (
            (i < 1000 and (i % 100 == 0))
            or (i < 10000 and (i % 500) == 0)
            or (i % 1000 == 0)
        ):
            torch.save(nn.state_dict(), "{dir}/model.{i}.pt".format(i=i, dir=args.name))
        (n, result) = trainer.td_episode(i)
        match result:
            case -3:
                results[0] += 1
            case -2:
                results[1] += 1
            case -1:
                results[2] += 1
            case 1:
                results[3] += 1
            case 2:
                results[4] += 1
            case 3:
                results[5] += 1
            case _:
                assert False
        print(
            results,
            n,
            (
                -3 * results[0]
                - 2 * results[1]
                - results[2]
                + results[3]
                + 2 * results[4]
                + 3 * results[5]
            )
            / (i + 1),
        )

    print(results)
    torch.save(
        nn.state_dict(),
        "{dir}/model.{i}.final.pt".format(i=args.iterations, dir=args.name),
    )
