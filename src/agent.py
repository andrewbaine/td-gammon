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
    def __init__(self, nn, move_tensors, encoder, device=torch.device("cuda")):
        self.device = device
        self.nn = network.with_backgammon_utility(nn, device=device)
        self.encoder = encoder
        self.move_tensors = move_tensors

    def evaluate(self, state):
        (board, player_1, _) = state
        te = self.encoder.encode(board, player_1)
        return self.nn(te)

    def next(self, state):
        (board, player_1, _) = state
        move_vectors = self.move_tensors.compute_move_vectors(state)
        next_states = torch.add(move_vectors, board)
        tesauro_encoded = self.encoder.encode(next_states, not player_1)
        vs = self.nn(tesauro_encoded)
        index = (torch.argmax if player_1 else torch.argmin)(vs)
        v_next = vs[index]
        board_next = next_states[index]
        return (v_next, board_next)

    def decide_action(self, state):
        (board, player_1, dice) = state
        board = torch.tensor(board, dtype=torch.float, device=self.device)
        state = (board, player_1, dice)

        (moves, move_vectors) = self.move_tensors.compute_moves(state)
        next_states = torch.add(move_vectors, board)
        tesauro_encoded = self.encoder.encode(next_states, not player_1)
        vs = self.nn(tesauro_encoded)
        index = (torch.argmax if player_1 else torch.argmin)(vs)
        move = [int(x) for x in moves[index].tolist()]
        move = list(reversed(sorted(zip(move[::3], move[1::3]))))
        return move
