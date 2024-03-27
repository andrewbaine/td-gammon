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
    tensor = tesauro.observe(s)
    v = nn(tensor).tolist()
    expected = [
        0.3743259906768799,
        0.5034601092338562,
        0.4764396548271179,
        0.43971505761146545,
    ]
    assert v == expected


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
    observe = tesauro.observe
    trainer = model.Trainer(b, nn, observe)

    v = nn(observe(s))

    v1 = nn(observe(s1))

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

    trainer.train(v1.dot(network.utility_tensor()).item(), s)

    assert nn(observe(s)).tolist() == [
        0.37418854236602783,
        0.5033813714981079,
        0.47652003169059753,
        0.43986862897872925,
    ]
    assert nn(observe(s1)).tolist() == [
        0.3717195987701416,
        0.5024778246879578,
        0.4750007390975952,
        0.43825823068618774,
    ]


def test_evaluations_of_single_game_play():
    random_seed = 42
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    torch.manual_seed(random_seed)
    nn = network.layered(198, 20, 4)
    bck = backgammon_env.Backgammon()
    trainer = model.Trainer(bck, nn, tesauro.observe)

    for i in range(5):
        trainer.td_episode(i)

    s = bck.s0(player_1=True)
    evaluations = []

    dice = None
    while True:
        (d1, d2) = tuple(torch.randint(1, 7, (2,)).tolist())
        if d1 != d2:
            dice = (d1, d2)
            break

    rolls = []
    while True:
        evaluations.append(trainer.v(s).item())
        if bck.done(s):
            break
        else:
            rolls.append(dice)
            move = trainer.best(s, dice)
            s = bck.next(s, move)
            dice = tuple(torch.randint(1, 7, (2,)).tolist())

    assert rolls == [
        (5, 2),
        (1, 2),
        (5, 5),
        (2, 6),
        (1, 4),
        (1, 5),
        (2, 3),
        (1, 6),
        (2, 5),
        (4, 4),
        (6, 2),
        (3, 3),
        (1, 4),
        (6, 6),
        (2, 6),
        (2, 4),
        (3, 5),
        (6, 4),
        (4, 1),
        (1, 6),
        (6, 3),
        (5, 5),
        (3, 2),
        (1, 5),
        (2, 3),
        (5, 1),
        (4, 5),
        (4, 4),
        (5, 1),
        (3, 5),
        (2, 1),
        (1, 1),
        (2, 3),
        (3, 6),
        (6, 1),
        (5, 1),
        (5, 5),
        (4, 2),
        (3, 3),
        (2, 3),
        (3, 1),
        (5, 5),
        (4, 3),
        (1, 3),
        (6, 6),
        (4, 4),
        (1, 6),
        (3, 3),
        (6, 4),
        (2, 4),
        (4, 3),
        (2, 4),
        (4, 3),
        (2, 3),
        (2, 6),
        (1, 2),
        (5, 3),
        (3, 2),
        (2, 6),
        (5, 6),
        (5, 2),
        (5, 5),
        (3, 2),
        (3, 3),
        (3, 5),
        (3, 1),
        (4, 6),
        (3, 4),
        (6, 3),
    ]

    assert evaluations == [
        1.5599075555801392,
        1.5657240152359009,
        1.5570687055587769,
        1.5735793113708496,
        1.5673117637634277,
        1.5806893110275269,
        1.566693902015686,
        1.5821315050125122,
        1.5677988529205322,
        1.578926682472229,
        1.5635840892791748,
        1.5499292612075806,
        1.5379986763000488,
        1.555314064025879,
        1.55019211769104,
        1.5632684230804443,
        1.5594894886016846,
        1.5647038221359253,
        1.5505921840667725,
        1.552159070968628,
        1.547096848487854,
        1.5598493814468384,
        1.5556155443191528,
        1.559731125831604,
        1.5500237941741943,
        1.5551419258117676,
        1.5503449440002441,
        1.5582256317138672,
        1.5524635314941406,
        1.5568782091140747,
        1.5465306043624878,
        1.552985429763794,
        1.5479785203933716,
        1.556337833404541,
        1.553457260131836,
        1.5596177577972412,
        1.5547544956207275,
        1.5509626865386963,
        1.5435014963150024,
        1.552736520767212,
        1.5468378067016602,
        1.5526933670043945,
        1.5482170581817627,
        1.5508335828781128,
        1.5407856702804565,
        1.5536787509918213,
        1.5506591796875,
        1.5469766855239868,
        1.543267846107483,
        1.5327744483947754,
        1.529687523841858,
        1.5435786247253418,
        1.540610671043396,
        1.5359041690826416,
        1.5327839851379395,
        1.5529522895812988,
        1.5458863973617554,
        1.5460877418518066,
        1.5449124574661255,
        1.562580943107605,
        1.5654520988464355,
        1.5738189220428467,
        1.5793131589889526,
        1.5876907110214233,
        1.5778114795684814,
        1.5862884521484375,
        1.5838605165481567,
        1.5878925323486328,
        1.5823419094085693,
        1.586571216583252,
    ]
