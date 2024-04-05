import backgammon
import torch

import backgammon_env


def test_game_play():
    torch.random.manual_seed(1)
    bck = backgammon_env.Backgammon(lambda: torch.randint(1, 7, (1,)).item())
    state = bck.s0()
    (board, player_1, dice) = state
    tensor_board = torch.tensor(board, dtype=torch.float)
    while True:
        (board, player_1, dice) = state
        print(backgammon.to_str(board))
        print(dice)

        done = bck.done(state)
        if done:
            break
        else:
            moves = bck.available_moves((board, player_1, dice))
            if moves:
                i = torch.randint(0, len(moves), (1,)).item()
                move = moves[i]
            else:
                move = []
            state = bck.next(state, move)


if __name__ == "__main__":
    test_game_play()
