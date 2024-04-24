import torch

from typing import Tuple


class Agent:
    def next(
        self, board: torch.Tensor, dice: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert False


class RandomAgent(Agent):
    def __init__(self, move_tensors):
        self.move_tensors = move_tensors

    def next(self, board, dice):
        move_vectors = self.move_tensors.compute_move_vectors(board, dice)
        next_states = torch.add(move_vectors, board)
        (m, n) = next_states.size()
        assert n == 27

        i = torch.randint(0, m, (1,))[0]
        return (0.5, next_states[i])


class OnePlyAgent(Agent):
    def __init__(self, nn, move_tensors):
        self.nn = nn
        self.move_tensors = move_tensors

    def evaluate(self, board):
        (_, n) = board.size()
        assert n == 27
        return self.nn(board)

    def next(self, board: torch.Tensor, dice):
        (m, n) = board.size()
        assert m > 0
        assert n == 27
        move_vectors = self.move_tensors.compute_move_vectors(board, dice)
        next_states = torch.add(move_vectors, board)
        (_, n) = next_states.size()
        assert n == 27
        utilities = self.evaluate(next_states)
        player_bit = board[:, [26]]
        us = 2 * player_bit - 1
        us = us * utilities

        index = torch.argmax(us)
        utility_next = utilities[index]
        board_next = next_states[index].unsqueeze(0)
        assert utility_next.size() == (m,), utility_next.size()
        assert board_next.size() == (m, 27), board_next.size()
        return (utility_next, board_next)
