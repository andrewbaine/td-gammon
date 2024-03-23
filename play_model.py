import td_gammon
import torch
import torch.nn as nn
import backgammon_env

layers = [198, 40, 4]


good = td_gammon.Network(*layers)
good.load_state_dict(torch.load("model.800.pt"))

bad = td_gammon.Network(*layers)
bad.load_state_dict(torch.load("model.0.pt"))


bck = backgammon_env.Backgammon()
observer = backgammon_env.Teasoro198()

import head_to_head

results = head_to_head.trial(
    (good, observer),
    (bad, observer),
    cb=lambda results: print(
        results, -2 * results[0] + -1 * results[1] + results[2] + 2 * results[3]
    ),
)
