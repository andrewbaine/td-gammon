import argparse

import torch

import backgammon
import backgammon_env
import model
import network
import tesauro
import policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=40)
    parser.add_argument("--model", required=True)
    parser.add_argument("--encoding", choices=["tesauro198"], required=True)
    args = parser.parse_args()

    bck = backgammon_env.Backgammon()
    observe = None

    match args.encoding:
        case "tesauro198":
            observe = tesauro.observe
        case _:
            assert False
    assert observe is not None

    t = observe(bck.s0(player_1=True))
    layers = [t.size()[0], args.hidden, 4]

    nn = network.layered(*layers)
    nn.load_state_dict(torch.load(args.model))
    nn = network.with_utility(nn)
    policy = policy.Policy_1_ply(bck, observe, nn)

    with torch.no_grad():

        dice = backgammon.first_roll()
        (d1, d2) = dice
        human_first = d1 < d2

        state = bck.s0()
        while True:
            (board, player_1) = state
            b = [x for x in board]
            if not human_first:
                backgammon.invert(b)
            print(backgammon.to_str(b, player_1_color=backgammon.Color.Dark))
            print("Dice:", dice)
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
                        print(
                            ("human" if not player_1 else "robot") + " won by 2 points"
                        )
                    case _:
                        raise Exception("unexpected")
                break
            humans_turn = player_1 and human_first or (not player_1 and not human_first)
            if humans_turn:
                move = None
                moves = bck.available_moves(state, dice)
                if moves:
                    input("press [return] to see your options")
                    for i, move in enumerate(moves):
                        print(i, move)
                    selection = None
                    while True:
                        try:
                            selection = int(input("Select move: "))
                            if -1 < selection and selection < len(moves):
                                break
                        except:
                            pass
                    assert selection is not None
                    move = moves[selection]
                    by_the_way = policy.choose_action((board, player_1, dice))
                else:
                    by_the_way = None
                    input("sorry, you can't move, press [return]")
            else:
                input("press enter to see computer move")
                (board, player_1) = state
                move = policy.choose_action((board, player_1, dice))
                by_the_way = None
            print("human" if humans_turn else "computer", "played", move)
            if by_the_way:
                print("by the way, computer liked", by_the_way)
            state = bck.next(state, move)
            dice = backgammon.roll()
