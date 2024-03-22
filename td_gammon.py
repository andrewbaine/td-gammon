import torch.nn as nn
import itertools

utility_tensor = (2, 1, -1, -2)


class Network(nn.Sequential):
    # http://incompleteideas.net/book/first/ebook/node87.html
    def __init__(self, *layers):
        super().__init__(
            *list(
                itertools.chain(
                    *[
                        (
                            [nn.Linear(n, layers[i + 1]), nn.Sigmoid()]
                            if i < (len(layers) - 1)
                            else []
                        )
                        for (i, n) in enumerate(layers)
                    ]
                )
            )
        )
