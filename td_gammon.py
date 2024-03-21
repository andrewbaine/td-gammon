import torch.nn as nn
import itertools


class Network(nn.Sequential):
    # http://incompleteideas.net/book/first/ebook/node87.html
    def __init__(self, in_features, hidden_features, n_hidden_layers):
        super().__init__(
            *(
                [nn.Linear(in_features, hidden_features), nn.Sigmoid()]
                + list(
                    itertools.chain(
                        *[
                            [nn.Linear(hidden_features, hidden_features), nn.Sigmoid()]
                            for _ in range(n_hidden_layers)
                        ]
                    )
                )
                + [nn.Linear(hidden_features, 4)]
            ),
        )
