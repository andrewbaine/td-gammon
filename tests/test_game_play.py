from itertools import zip_longest
import os
import random

from pytest import approx
import torch

from td_gammon import backgammon_env, done_check, move_vectors

from . import baine_encoding, slow_but_right, tesauro


def normalize(moves):
    return list(reversed(sorted(tuple(reversed(sorted(x))) for x in moves)))


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

    n = int(os.environ.get("N_GAMES", default="1"))
    move_tensors = move_vectors.MoveTensors()

    for i in range(n):
        state = bck.s0()
        (board, player_1, dice) = state
        tensor_board = torch.tensor([board + [1 if player_1 else 0]], dtype=torch.float)
        while True:
            (board, player_1, dice) = state
            assert [int(x) for x in tensor_board.tolist()[0][:-1]] == board

            t = encoder.encode(tensor_board[:, :-1], player_1).tolist()[0]
            t2 = slow_but_right.tesauro_encode(state)

            baine_encoded = slow_but_right.simple_baine_encoding_step_1(board[1:25])
            be2 = [
                int(x)
                for x in enc.encode_step_1(tensor_board[:, :-1]).squeeze().tolist()
            ]

            assert baine_encoded == be2

            be3 = [
                int(x)
                for x in enc.encode_step_2(enc.encode_step_1(tensor_board[:, :-1]))
                .squeeze()
                .tolist()
            ]
            be3_slow = slow_but_right.simple_baine_encoding_step_2(
                baine_encoded, min=1, max=4
            )

            assert be3 == be3_slow
            assert len(t) == len(t2)
            assert t2 == approx(t)
            done = bck.done(state)
            done_slow_but_right = sbr.done(state)
            dc_done = dc.check(tensor_board)

            assert done_slow_but_right == done, "done_slow_but_right done should agree"
            assert dc_done == done, "done_check done should agree"
            if done:
                break
            else:
                moves = bck.available_moves(state)
                vectors = move_tensors.compute_move_vectors(tensor_board, dice)
                next_states = vectors + tensor_board
                encoded_next_states = enc.encode(
                    next_states[:, :-1], not player_1
                ).tolist()
                other_way = [
                    enc.encode(torch.tensor(x)[:-1], not player_1).tolist()[0]
                    for x in next_states.tolist()
                ]
                one_final_way = [
                    slow_but_right.simple_baine_encoding((x, not player_1, (0, 0)))
                    for x in next_states[:, :-1].tolist()
                ]
                for x, y, z in zip_longest(
                    encoded_next_states, other_way, one_final_way
                ):
                    assert x == approx(y)
                    assert x == approx(z)

                ### the states we can end up using vectors
                s = set()
                vv = dict()
                for v in vectors + tensor_board:
                    key = tuple(int(x) for x in v[:-1].tolist())
                    vv[key] = v
                    s.add(key)

                ### should be the same as
                ### the states we can end up using moves
                s2 = set()
                for m in moves:
                    (b, _, _) = bck.next((board, player_1, dice), m)
                    key = tuple(b)
                    assert key in s
                    s2.add(key)
                for x in s:
                    assert x in s2

                mmm = normalize(sbr.compute_moves(state))
                assert mmm == moves

                if moves:
                    i = torch.randint(0, len(moves), (1,))[0]
                    move = moves[i]
                else:
                    move = ()

                state = bck.next(state, move)
                (x, _, _) = state
                tensor_board = torch.unsqueeze(vv[tuple(x)], dim=0)


if __name__ == "__main__":
    test_game_play()
