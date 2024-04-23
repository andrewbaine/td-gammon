import encoders
import backgammon
import torch
import slow_but_right


def test_starting_board():
    t = encoders.TesauroOneHot()
    board = torch.tensor([backgammon.make_board()], dtype=torch.float)
    encoded = t(board).tolist()[0]
    tesauro_encoded = slow_but_right.tesauro_encode(
        (backgammon.make_board(), True, (0, 0))
    )
    assert encoded == tesauro_encoded[0 : (8 * 24 + 2)]
