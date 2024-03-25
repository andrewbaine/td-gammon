import torch

import backgammon
import backgammon_env
import model
import network


def trial(a, b, games=100, cb=None):
    bck = backgammon_env.Backgammon()
    (n1, observer_1) = a
    (n2, observer_2) = b
    results = [0, 0, 0, 0]
    n1 = network.with_utility(n1)
    n2 = network.with_utility(n2)
    with torch.no_grad():
        for _ in range(games):
            dice = backgammon.first_roll()
            state = bck.s0()

            while True:
                (_, player_1) = state
                (n, observer) = (n1, observer_1) if player_1 else (n2, observer_2)
                done = bck.done(state)
                if done is not None:
                    match done:
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
                    break
                move = model.best(bck, observer, state, dice, n)
                state = bck.next(state, move)
                dice = backgammon.roll()
            if cb is not None:
                cb(results)
    return results
