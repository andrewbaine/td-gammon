import torch

import backgammon
import backgammon_env
import model
import network
import tesauro

layers = [198, 40, 4]

nn = network.layered(*layers)
nn.load_state_dict(torch.load("model.3900.pt"))
nn = network.with_utility(nn)

i_am_player_1 = int(input("Are you first? 1/0 for Yes/No: "))


bck = backgammon_env.Backgammon()
observer = tesauro.Tesauro198()

state = bck.s0()

with torch.no_grad():
    while True:
        (board, player_1) = state
        my_turn = player_1 if i_am_player_1 else (not player_1)
        print(backgammon.to_str(board))
        if my_turn:
            d1 = int(input("Enter d1: "))
            d2 = int(input("Enter d2: "))
            move = model.best(bck, observer, state, (d1, d2), nn)
            print("Rolled", (d1, d2), "; played ", move)
            state = bck.next(state, move)
        else:
            move = None
            line = input(("Enter move: "))
            if line:
                tokens = [int(x) for x in line.split()]
                move = []
                i = 0
                while i < len(tokens):
                    move.append((tokens[i], tokens[i + 1]))
                    i += 2
            state = bck.next(state, move)
