import torch

import backgammon
import backgammon_env
import model


def trial(a, b, games=100, cb=None):
    bck = backgammon_env.Backgammon()
    [trainer_1, trainer_2] = [model.Trainer(bck, n, observe) for (n, observe) in [a, b]]
    results = [0, 0, 0, 0]
    with torch.no_grad():
        for _ in range(games):
            dice = backgammon.first_roll()
            state = bck.s0()

            while True:
                (_, player_1) = state
                trainer = (trainer_1) if player_1 else (trainer_2)
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
                move = trainer.best(state, dice)
                state = bck.next(state, move)
                dice = backgammon.roll()
            if cb is not None:
                cb(results)
    return results
