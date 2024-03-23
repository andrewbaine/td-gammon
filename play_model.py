import torch
import backgammon_env
import network

layers = [198, 40, 4]


good = network.layered(*layers)
good.load_state_dict(torch.load("model.3000.pt"))

bad = network.layered(*layers)
bad.load_state_dict(torch.load("model.2500.pt"))


bck = backgammon_env.Backgammon()
observer = backgammon_env.Teasoro198()

import head_to_head

results = head_to_head.trial(
    (good, observer),
    (bad, observer),
    cb=lambda results: print(
        results,
        (-2 * results[0] + -1 * results[1] + results[2] + 2 * results[3])
        / (sum(results)),
    ),
)
