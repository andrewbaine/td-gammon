import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def observe(state):
    tensor = [0 for _ in range(198)]
    (board, player_1, dice) = state

    checker_count_1 = 15.0
    checker_count_2 = 15.0
    t = 0
    for i, pc in enumerate(board):
        if i == 0:
            assert pc <= 0
            checker_count_2 += pc
            tensor[192] = pc / -2.0
        elif i < 25:
            if pc > 0:
                checker_count_1 -= pc
                tensor[t] = 1
                if pc > 1:
                    tensor[t + 1] = 1
                    if pc > 2:
                        tensor[t + 2] = 1
                        if pc > 3:
                            tensor[t + 3] = (pc - 3) / 2
            elif pc < 0:
                checker_count_2 += pc
                tensor[t + 4] = 1
                if pc < -1:
                    tensor[t + 5] = 1
                    if pc < -2:
                        tensor[t + 6] = 1
                        if pc < -3:
                            tensor[t + 7] = (-3 - pc) / 2
            t += 8
        elif i == 25:
            assert pc >= 0
            checker_count_1 -= pc
            tensor[193] = pc / 2.0
        else:
            assert False

    tensor[194] = checker_count_1 / 15.0
    tensor[195] = checker_count_2 / 15.0
    tensor[196] = 1 if player_1 else 0
    tensor[197] = 0 if player_1 else 1
    return torch.tensor(tensor, device=device)


def tensor():
    return torch.tensor([0 for _ in range(198)], dtype=torch.float)
