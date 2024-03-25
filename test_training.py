import torch

import backgammon_env
import model
import network
import tesauro


def test_predictability_of_network():
    random_seed = 42

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    torch.manual_seed(random_seed)

    nn = network.layered(198, 40, 4)

    b = backgammon_env.Backgammon()
    s = b.s0(player_1=True)
    observer = tesauro.Tesauro198()
    o = observer.observe(s)
    t = torch.tensor(o)
    v = nn(t)
    expected = [
        0.3743259906768799,
        0.5034601092338562,
        0.4764396548271179,
        0.43971505761146545,
    ]
    assert v.tolist() == expected


def test_predictability_of_network_after_train():
    random_seed = 42

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    torch.manual_seed(random_seed)

    nn = network.layered(198, 40, 4)

    b = backgammon_env.Backgammon()
    s = b.s0(player_1=True)
    m = ((8, 5), (6, 5))
    s1 = b.next(s, m)
    observer = tesauro.Tesauro198()
    trainer = model.Trainer(nn)

    o = observer.observe(s)
    t = torch.tensor(o)
    v = nn(t)

    o1 = observer.observe(s1)
    t1 = torch.tensor(o1)
    v1 = nn(t1)

    assert v.tolist() == [
        0.3743259906768799,
        0.5034601092338562,
        0.4764396548271179,
        0.43971505761146545,
    ]
    assert v1.tolist() == [
        0.37185654044151306,
        0.5025563836097717,
        0.47492071986198425,
        0.43810516595840454,
    ]

    trainer.train(v1.dot(torch.tensor([-2, -1, 1, 2], dtype=torch.float)).item(), o)

    assert nn(t).tolist() == [
        0.37418854236602783,
        0.5033813714981079,
        0.47652003169059753,
        0.43986862897872925,
    ]
    assert nn(t1).tolist() == [
        0.3717195987701416,
        0.5024778246879578,
        0.4750007390975952,
        0.43825823068618774,
    ]
