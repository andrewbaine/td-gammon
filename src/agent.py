import torch

import network


class Agent:
    pass


class OnePlyAgent(Agent):
    def __init__(self, nn, move_tensors, encoder, device=torch.device("cuda")):
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
