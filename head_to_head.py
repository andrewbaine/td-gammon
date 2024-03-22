import torch
import backgammon
import network
import model
import backgammon_env


def trial(a, b, games=100, cb=None):
    bck = backgammon_env.Backgammon()
    (n1, observer_1) = a
    (n2, observer_2) = b
    results = [0, 0, 0, 0]
    n1 = network.with_utility(n1)
    n2 = network.with_utility(n2)
    with torch.no_grad():
        for i in range(games):
            dice = backgammon.first_roll()
            (d1, d2) = dice
            player_1_is_n1 = d1 > d2
            (p1, p2) = (
                ((n1, observer_1), (n2, observer_2))
                if player_1_is_n1
                else ((n2, observer_2), (n1, observer_1))
            )
            state = bck.s0()

            while True:
                (board, player_1) = state
                (n, observer) = p1 if player_1 else p2
                done = bck.done(state)
                if done is not None:
                    match done:
                        case -2:
                            results[0 if player_1_is_n1 else 3] += 1
                        case -1:
                            results[1 if player_1_is_n1 else 2] += 1
                        case 1:
                            results[2 if player_1_is_n1 else 1] += 1
                        case 2:
                            results[3 if player_1_is_n1 else 0] += 1
                        case _:
                            raise Exception("unexpected")
                    break
                move = model.best(bck, observer, state, dice, n)
                state = bck.next(state, move)
                dice = backgammon.roll()
            if cb is not None:
                cb(results)
    return results
