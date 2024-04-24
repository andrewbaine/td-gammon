import torch


class ElibilityTrace:
    def __init__(self, nn, α, λ):
        self.α = α
        self.λ = λ
        self.nn = nn
        self.et = [
            (w, torch.zeros_like(w, requires_grad=False)) for w in nn.parameters()
        ]

    def update(self, v_predicted, v_actual):
        self.nn.zero_grad()
        v_predicted.backward()
        with torch.no_grad():
            δ = v_actual - v_predicted  # td error
            αδ = self.α * δ  # learning rate * td error

            for w, e in self.et:
                e.mul_(self.λ)
                e.add_(w.grad)
                w.add_(torch.mul(e, αδ))
