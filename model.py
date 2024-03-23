import torch

import torch.nn as nn
import backgammon

import network


class Trainer:
    def __init__(self, model, α=0.10, λ=0.7):
        self.α = α
        self.λ = λ
        self.nn = network.with_utility(model)
        self.eligibility_trace = [
            torch.zeros_like(w, requires_grad=False) for w in self.nn.parameters()
        ]

    def v(self, observation):
        return self.nn(observation)

    def train(self, v_next, observation):
        self.nn.zero_grad()
        v = self.nn(observation)
        v.backward()
        with torch.no_grad():
            δ = v_next - v.item()  # td error
            αδ = self.α * δ  # learning rate * td error
            et = self.eligibility_trace
            for i, weights in enumerate(self.nn.parameters()):
                if weights.grad is None:
                    raise Exception("we need grad")
                et[i] = torch.add(torch.mul(et[i], self.λ), weights.grad)
                weights.add_(torch.mul(et[i], αδ))
            # utility function is always -2x1 - x2 + x3 + 2x4
            self.nn.utility.weight = nn.Parameter(
                torch.tensor([x for x in (-2, -1, 1, 2)], dtype=torch.float)
            )


def best(bck, observer, gamestate, dice, network):
    (_, player_1) = gamestate
    moves = bck.allowed_moves = bck.available_moves(gamestate, dice)
    best = None
    best_move = None
    for move in moves:
        s = bck.next(gamestate, move)
        tensor = observer.observe(s)
        y = network(tensor).item()
        if best is None or ((y < best) if player_1 else (y > best)):
            best = y
            best_move = move
    return best_move


def td_episode(bck, observer, trainer, num):
    # https://medium.com/clique-org/td-gammon-algorithm-78a600b039bb
    state = bck.s0(player_1=(num % 2 == 0))
    dice = backgammon.first_roll()
    #   (board, _) = state
    #   print(backgammon.to_str(board))

    t = 0
    while True:
        o = observer.observe(state)
        done = bck.done(state)
        if done:
            trainer.train(done, o)
            return (t, done)
        else:
            with torch.no_grad():
                best_move = best(bck, observer, state, dice, trainer.nn)
            state = bck.next(state, best_move)
            next_observation = observer.observe(state)
            v_next = trainer.v(next_observation).item()
            trainer.train(v_next, o)
            dice = backgammon.roll()
        t += 1


import backgammon_env

if __name__ == "__main__":
    layers = [198, 40, 4]
    net = network.layered(*layers)
    trainer = Trainer(net)
    observer = backgammon_env.Teasoro198()

    n_episodes = 100000
    results = [0, 0, 0, 0]
    lengths = []
    bck = backgammon_env.Backgammon()
    for i in range(0, n_episodes):
        if i > 1000:
            trainer.α = 0.1
        if i > 5000:
            trainer.α = 0.05
        if (
            (i < 1000 and (i % 100 == 0))
            or (i < 10000 and (i % 500) == 0)
            or (i % 1000 == 0)
        ):
            torch.save(net.state_dict(), "model.{i}.pt".format(i=i))
        (n, result) = td_episode(bck, observer, trainer, i)
        lengths.append(n)
        match result:
            case -2:
                results[0] += 1
            case -1:
                results[1] += 1
            case 1:
                results[2] += 1
            case 2:
                results[3] += 1
            case _:
                raise Exception("unexpected")
        print(
            results,
            n,
            (-2 * results[0] - results[1] + results[2] + 2 * results[3]) / (i + 1),
        )
        assert (
            trainer.nn.utility(torch.tensor([1, 2, 3, 4], dtype=torch.float)).item()
            == 7.0
        )

    print(lengths, results)
    torch.save(net.state_dict(), "model.final.pt")
