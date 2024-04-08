import os
import backgammon
import torch
import read_move_tensors

import backgammon_env

import done_check


def translate(xs):
    ys = list(
        reversed(
            sorted(
                [
                    (tuple(reversed(sorted(zip(x[::3], x[1::3])))), i)
                    for (i, x) in enumerate(xs.tolist())
                ]
            )
        )
    )
    return ys


def normalize(moves):
    return list(reversed(sorted(tuple(reversed(sorted(x))) for x in moves)))


import random
import slow_but_right


def test_game_play():
    default_seed = random.randint(0, 0xFFFFFFFFFFFFFFFF)
    seed = int(os.environ.get("SEED", default=default_seed))
    print("seed for reuse", seed)
    torch.random.manual_seed(seed)
    bck = backgammon_env.Backgammon(lambda: torch.randint(1, 7, (1,)).item())

    sbr = slow_but_right.MoveComputer()
    dc = done_check.Donecheck()

    dir = os.environ.get("MOVES_TENSORS", default="move_tensors/current")
    n = int(os.environ.get("N_GAMES", default="10"))
    move_tensors = read_move_tensors.MoveTensors(dir=dir)

    for i in range(n):
        state = bck.s0()
        (board, player_1, dice) = state
        tensor_board = torch.tensor(board, dtype=torch.float)
        while True:
            (board, player_1, dice) = state
            print(backgammon.to_str(board))
            print("player", 1 if player_1 else 2, dice)
            assert tuple(tensor_board.tolist()) == board

            done = bck.done(state)
            done_slow_but_right = sbr.done(state)
            dc_done = dc.check(tensor_board)

            assert done_slow_but_right == done, "done_slow_but_right done should agree"
            assert dc_done == done, "done_check done should agree"
            if done:
                break
            else:
                moves = bck.available_moves(state)
                moves = normalize(moves)
                mm = move_tensors.compute_moves((tensor_board, player_1, dice))
                v = move_tensors.compute_move_vectors((tensor_board, player_1, dice))

                translated_moves = translate(mm)
                d = {}
                for move, index in translated_moves:
                    d[move] = index
                simplified_moves = [v for (v, _) in translated_moves]
                assert simplified_moves == moves or (
                    simplified_moves == [()] and moves == []
                )

                if moves:
                    i = torch.randint(0, len(moves), (1,)).item()
                    move = moves[i]
                else:
                    move = ()
                vector_move = v[d[move]]

                state = bck.next(state, move)
                tensor_board = tensor_board + vector_move


if __name__ == "__main__":
    test_game_play()
