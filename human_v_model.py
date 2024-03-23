import torch
import backgammon
import network
import backgammon_env
import model

layers = [198, 40, 4]

n = network.layered(*layers)
n.load_state_dict(torch.load("model.700.pt"))
n = network.with_utility(n)


bck = backgammon_env.Backgammon()
observer = backgammon_env.Teasoro198()

with torch.no_grad():

    dice = backgammon.first_roll()
    (d1, d2) = dice
    human_first = d1 < d2

    state = bck.s0()
    while True:
        (board, player_1) = state
        done = bck.done(state)
        if done is not None:
            match done:
                case -2:
                    print(("human" if player_1 else "robot") + " won by 2 points")
                case -1:
                    print(("human" if player_1 else "robot") + " won 1 point")
                case 1:
                    print(("human" if not player_1 else "robot") + " won 1 point")
                case 2:
                    print(("human" if not player_1 else "robot") + " won by 2 points")
                case _:
                    raise Exception("unexpected")
            break
        if player_1 and human_first or (not player_1 and not human_first):
            if human_first:
                print(backgammon.to_str(board))
            else:
                backgammon.invert(board)
                print(backgammon.to_str(board))
                backgammon.invert(board)
            print("Roll:", dice)
            move = None
            moves = bck.available_moves(state, dice)
            if moves:
                for i, move in enumerate(moves):
                    print(i, move)
                i = int(input("Select move: "))
                move = moves[i]
            state = bck.next(state, move)
        else:
            move = model.best(bck, observer, state, dice, network)
            print("Rolled", (d1, d2), "; played ", move)
            state = bck.next(state, move)
        dice = backgammon.roll()
