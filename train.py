import backgammon_env
import network
from model import Trainer, td_episode
import tesauro
import torch

if __name__ == "__main__":
    layers = [198, 40, 4]
    net = network.layered(*layers)
    trainer = Trainer(net)
    observer = tesauro.Tesauro198()

    n_episodes = 300000
    results = [0, 0, 0, 0]
    lengths = []
    bck = backgammon_env.Backgammon()
    for i in range(0, n_episodes):
        if (
            (i < 1000 and (i % 100 == 0))
            or (i < 10000 and (i % 500) == 0)
            or (i % 1000 == 0)
        ):
            torch.save(net.state_dict(), "model.{i}.pt".format(i=i))
        (n, result) = td_episode(bck, observer, trainer, i)
        lengths.append(n)
        match result:
            case -2:
                results[0] += 1
            case -1:
                results[1] += 1
            case 1:
                results[2] += 1
            case 2:
                results[3] += 1
            case _:
                assert False
        print(
            results,
            n,
            (-2 * results[0] - results[1] + results[2] + 2 * results[3]) / (i + 1),
        )
        assert (
            trainer.nn.utility(torch.tensor([1, 2, 3, 4], dtype=torch.float)).item()
            == 7.0
        )

    print(lengths, results)
    torch.save(net.state_dict(), "model.final.pt")
