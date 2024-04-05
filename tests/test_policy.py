import torch

import backgammon
import backgammon_env
import network
import policy
import tes as tesauro


def test_1_ply():
    random_seed = 1
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    torch.manual_seed(random_seed)
    observe = tesauro.observe

    # load a trained model
    # create 3 policies
    # play a game
    # compare the 3 policies

    hidden = 83
    layers = [198, hidden, 4]
    softmax = True
    nn = network.layered(*layers, softmax=softmax)

    dir = "tesauro-2024-03-27"
    file = "model.30000.pt"

    nn.load_state_dict(torch.load("{dir}/{file}".format(dir=dir, file=file)))
    nn = network.with_utility(nn)

    bck = backgammon_env.Backgammon()
    policies = [
        policy.Policy_1_ply(bck, observe, nn),
        policy.Policy_2_ply_selective(bck, observe, nn),
    ]

    dice = None
    while True:
        (d1, d2) = tuple(torch.randint(1, 7, (2,)).tolist())
        if d1 != d2:
            dice = (d1, d2)
            break

    rolls = []
    selections = []
    s = bck.s0()
    (board, player_1_ignoree, dice_ignored) = s
    player_1 = True
    s = (board, player_1, dice)
    i = 0
    while True:
        i += 1
        if bck.done(s):
            break
        else:
            rolls.append(dice)
            moves = [p.choose_action((board, player_1, dice)) for p in policies]
            if moves[0] != moves[1]:
                print(backgammon.to_str(board))
                print("player 1", player_1)
                print(moves[0])
                print(moves[1])
            selections.append(moves)
            move = moves[-1]
            (board, player_1, dice_ignored) = bck.next(s, move)
            dice = tuple(torch.randint(1, 7, (2,)).tolist())
            s = (board, player_1, dice)

    assert selections == [
        [((13, 7), (8, 7)), ((13, 7), (8, 7))],
        [((13, 9), (13, 9), (9, 5), (9, 5)), ((13, 9), (13, 9), (9, 5), (9, 5))],
        [((13, 9), (13, 8)), ((13, 9), (13, 8))],
        [((24, 20), (8, 5)), ((24, 20), (8, 5))],
        [((6, 5), (6, 1)), ((6, 5), (6, 1))],
        [((25, 23), (25, 22)), ((25, 23), (25, 22))],
        [((9, 3), (6, 5)), ((9, 3), (6, 5))],
        [((25, 24), (13, 8)), ((25, 24), (13, 8))],
        [((25, 22), (24, 22)), ((25, 22), (24, 22))],
        [((8, 2), (5, 2)), ((8, 2), (5, 2))],
        [((24, 21), (21, 15)), ((24, 18), (18, 15))],
        [((13, 8), (13, 7)), ((13, 8), (13, 7))],
        [((22, 18), (3, 2)), ((22, 18), (3, 2))],
        [((25, 24), (25, 21)), ((25, 24), (25, 21))],
        [((22, 16), (16, 10), (10, 4), (8, 2)), ((22, 16), (16, 10), (10, 4), (8, 2))],
        [None, None],
        [((13, 10), (10, 4)), ((13, 7), (7, 4))],
        [((25, 22), (8, 3)), ((25, 22), (8, 3))],
        [((18, 15), (15, 14)), ((18, 15), (15, 14))],
        [((22, 16), (6, 3)), ((22, 16), (6, 3))],
        [((15, 11), (13, 9), (7, 3), (7, 3)), ((15, 11), (13, 9), (7, 3), (7, 3))],
        [None, None],
        [((11, 6), (9, 8)), ((11, 6), (9, 8))],
        [None, None],
        [((14, 11), (11, 7)), ((14, 10), (10, 7))],
        [((25, 24), (6, 4)), ((25, 24), (6, 4))],
        [((8, 7), (8, 4)), ((8, 7), (8, 4))],
        [((8, 5), (8, 2)), ((8, 5), (8, 2))],
        [((8, 3), (4, 3)), ((8, 3), (4, 3))],
        [((6, 5), (5, 4), (2, 1), (2, 1)), ((6, 5), (5, 4), (2, 1), (2, 1))],
        [((7, 3), (7, 3), (4, 0), (4, 0)), ((7, 3), (7, 3), (4, 0), (4, 0))],
        [((24, 18), (18, 14)), ((24, 18), (18, 14))],
        [((3, 2), (3, 0)), ((3, 2), (3, 0))],
        [((14, 13), (13, 7)), ((14, 8), (8, 7))],
        [((6, 5), (2, 0)), ((6, 5), (2, 0))],
        [((7, 2), (5, 4)), ((7, 2), (5, 4))],
        [((6, 2), (6, 2)), ((6, 2), (6, 2))],
        [((24, 19), (19, 13)), ((24, 18), (18, 13))],
        [((5, 1), (3, 1)), ((5, 1), (3, 1))],
        [((25, 19), (13, 8)), ((25, 19), (13, 8))],
        [((2, 0), (2, 0), (2, 0), (2, 0)), ((2, 0), (2, 0), (2, 0), (2, 0))],
        [((19, 15), (15, 10)), ((19, 14), (14, 10))],
        [((5, 0), (3, 0)), ((5, 0), (3, 0))],
        [((10, 9), (9, 4)), ((10, 5), (5, 4))],
        [((5, 0), (3, -3)), ((5, -1), (3, -2))],
        [((8, 5), (5, 2), (4, 1), (4, 1)), ((8, 5), (5, 2), (4, 1), (4, 1))],
        [((3, 2), (2, -3)), ((3, 2), (2, -3))],
        [((6, 0), (1, 0)), ((6, 0), (1, 0))],
        [((1, 0), (1, -2)), ((1, 0), (1, -2))],
    ]
