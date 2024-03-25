class Tesauro198:
    def __init__(self):
        super().__init__()

    def observe(self, state):
        (board, player_1) = state
        tensor = [0.0 for _ in range(198)]
        a = 15.0
        b = 15.0
        for i, pc in enumerate(board):
            if i == 0:
                if pc < 0:
                    tensor[192] = pc / -2.0
                elif pc > 0:
                    raise Exception("unexpected at i == 0")
            elif i < 25:
                t = 8 * (i - 1)
                for j in range(0, 8):
                    tensor[t + j] = 0.0
                if pc > 0:
                    a -= pc
                    tensor[t] = 1.0
                    if pc > 1:
                        tensor[t + 1] = 1.0
                        if pc > 2:
                            tensor[t + 2] = 1.0
                            if pc > 3:
                                tensor[t + 3] = (pc - 3.0) / 2.0
                elif pc < 0:
                    b += pc
                    tensor[t + 4] = 1.0
                    if pc < -1:
                        tensor[t + 5] = 1.0
                        if pc < -2:
                            tensor[t + 6] = 1.0
                            if pc < -3:
                                tensor[t + 7] = (-1 * pc - 3.0) / 2.0
                            pass
            elif i == 25:
                if pc < 0:
                    raise Exception("unexpected at i == 25")
                elif pc > 0:
                    tensor[193] = pc / 2.0
            else:
                raise Exception("impossible")

        tensor[194] = a / 15.0
        tensor[195] = b / 15.0
        tensor[196] = 1 if player_1 else 0
        tensor[197] = 0 if player_1 else 1
        return tensor
