import pytest
import os
import backgammon
import torch
import read_move_tensors
import tesauro

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

import baine_encoding


def test_game_play():
    encoder = tesauro.Encoder()
    enc = baine_encoding.Encoder()

    default_seed = random.randint(0, 0xFFFFFFFFFFFFFFFF)
    seed = int(os.environ.get("SEED", default=default_seed))
    print("seed for reuse", seed)
    torch.random.manual_seed(seed)
    bck = backgammon_env.Backgammon(lambda: torch.randint(1, 7, (1,)).item())

    sbr = slow_but_right.MoveComputer()
    dc = done_check.Donecheck()

    dir = os.environ.get("MOVES_TENSORS", default="var/move_tensors/current")
    n = int(os.environ.get("N_GAMES", default="10"))
    move_tensors = read_move_tensors.MoveTensors(dir=dir)

    for i in range(n):
        state = bck.s0()
        (board, player_1, dice) = state
        tensor_board = torch.tensor(board, dtype=torch.float)
        while True:
            (board, player_1, dice) = state
            assert [int(x) for x in tensor_board.tolist()] == board

            t = encoder.encode(tensor_board, player_1).tolist()
            t2 = slow_but_right.tesauro_encode(state)

            baine_encoded = slow_but_right.simple_baine_encoding_step_1(board)
            be2 = [int(x) for x in enc.encode_step_1(tensor_board).tolist()]

            #            print(backgammon.to_str(board))
            assert baine_encoded == be2

            be3 = [
                int(x)
                for x in enc.encode_step_2(enc.encode_step_1(tensor_board)).tolist()
            ]
            be3_slow = slow_but_right.simple_baine_encoding_step_2(
                baine_encoded, min=1, max=4
            )

            assert be3 == be3_slow
            assert len(t) == len(t2)
            for i in range(len(t)):
                if t2[i] != t[i]:
                    print(t)
                    print(t2)
                assert t2[i] == pytest.approx(t[i])
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
                (mm, vv) = move_tensors.compute_moves((tensor_board, player_1, dice))
                v = move_tensors.compute_move_vectors((tensor_board, player_1, dice))
                assert (v.unique(dim=0) == vv.unique(dim=0)).all()

                translated_moves = translate(mm)
                d = {}
                for move, index in translated_moves:
                    d[move] = index
                simplified_moves = [v for (v, _) in translated_moves]
                assert simplified_moves == moves or (
                    simplified_moves == [()] and moves == []
                )

                mmm = normalize(sbr.compute_moves(state))
                assert mmm == moves

                if moves:
                    i = torch.randint(0, len(moves), (1,)).item()
                    move = moves[i]
                else:
                    move = ()
                vector_move = vv[d[move]]

                state = bck.next(state, move)
                tensor_board = tensor_board + vector_move


if __name__ == "__main__":
    test_game_play()
