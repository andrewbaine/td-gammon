import os
import backgammon
import torch
import read_move_tensors

import backgammon_env


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


def test_game_play():
    torch.random.manual_seed(1)
    bck = backgammon_env.Backgammon(lambda: torch.randint(1, 7, (1,)).item())
    dir = os.environ.get("MOVES_TENSORS", default="move_tensors/current")
    n = int(os.environ.get("N_GAMES", default="10000"))
    move_tensors = read_move_tensors.MoveTensors(dir=dir)

    for i in range(3):
        state = bck.s0()
        (board, player_1, dice) = state
        tensor_board = torch.tensor(board, dtype=torch.float)
        while True:
            (board, player_1, dice) = state
            print(backgammon.to_str(board))
            print("player", 1 if player_1 else 2, dice)
            assert tuple(tensor_board.tolist()) == board

            done = bck.done(state)
            if done:
                break
            else:
                moves = bck.available_moves(state)
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
                print(mm)
                print(translate(mm))
                print(moves)

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
