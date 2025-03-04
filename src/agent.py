import random

import torch

import backgammon_env
import network


class Agent:
    pass


class RandomAgent(Agent):
    def __init__(self):
        self.bck = backgammon_env.Backgammon()

    def decide_action(self, state):
        ms = self.bck.available_moves(state)
        if not ms:
            return None
        i = random.randint(0, len(ms) - 1)
        return ms[i]


class OnePlyAgent(Agent):
    def __init__(self, nn, move_tensors, encoder, out):
        self.nn = nn
        match out:
            case 4:
                self.utility = network.utility_tensor()
            case 6:
                self.utility = network.backgammon_utility_tensor()
            case _:
                assert False
        self.encoder = encoder
        self.move_tensors = move_tensors

    def f(self, tes):
        vs = self.nn(tes)
        y = torch.softmax(vs, dim=1)
        y = torch.matmul(y, self.utility)
        return y

    def evaluate(self, state):
        (board, player_1, _) = state
        board = board.expand(size=(1, -1))
        te = self.encoder.encode(board, player_1)
        return self.f(te)

    def next(self, state):
        (board, player_1, _) = state
        move_vectors = self.move_tensors.compute_move_vectors(state)
        next_states = torch.add(move_vectors, board)
        tesauro_encoded = self.encoder.encode(next_states, not player_1)
        utilities = self.f(tesauro_encoded)
        index = (torch.argmax if player_1 else torch.argmin)(utilities)
        utility_next = utilities[index]
        board_next = next_states[index]
        return (utility_next, board_next)

    def decide_action(self, state):
        (board, player_1, dice) = state

        board = torch.tensor(
            board,
            dtype=torch.float,
        )
        state = (board, player_1, dice)

        (moves, move_vectors) = self.move_tensors.compute_moves(state)
        next_states = torch.add(move_vectors, board)
        te = self.encoder.encode(next_states, not player_1)
        utilities = self.f(te)

        index = (torch.argmax if player_1 else torch.argmin)(utilities)
        move = [int(x) for x in moves[index].tolist()]
        move = list(reversed(sorted(zip(move[::3], move[1::3]))))
        return move
