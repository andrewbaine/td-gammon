import base64
from collections import namedtuple


def pad(s):
    i = len(s) % 4
    while i > 0:
        i -= 1
        s = s + "="
    return s


def decode_position(position_id):
    bs = base64.b64decode(pad(position_id))
    assert len(bs) == 10
    arr = []
    board = [0 for x in range(26)]
    index = 1
    player = 1
    for b in bs:
        for i in range(0, 8):
            flag = 1 if b & (1 << i) else 0
            if not flag:
                index += player
                if index > 25:
                    player = -1
                    index = 24
            else:
                board[index] += player
            arr.append(flag)
    return board


Match = namedtuple(
    "Match",
    [
        "cube",
        "cube_owner",
        "dice_owner",
        "crawford",
        "gamestate",
        "turn",
        "double",
        "resign_flag",
        "dice",
        "match_length",
        "score",
    ],
)

from enum import Enum

GameState = Enum(
    "GameState", ["NO_GAME_STARTED", "PLAYING", "GAME_OVER", "RESIGNED", "OVER_BY_DROP"]
)

ResignFlag = Enum(
    "ResignFlag",
    [
        "NO_RESIGNATION",
        "SINGLE_GAME_OFFERED",
        "GAMMON_OFFERED",
        "BACKGAMMON_OFFERED",
    ],
)


def bit(bs, i):
    b = bs[i // 8]
    return 1 if (b & (1 << (i % 8))) else 0


def n(bs, start, end):
    assert start < end
    n = 0
    b = 1
    for i in range(start, end):
        n += bit(bs, i) * b
        b *= 2
    return n


def decode_match(match_id):
    bs = base64.b64decode(pad(match_id), validate=False)
    exp = n(bs, 0, 4)
    cube = 2**exp
    cube_owner_mask = n(bs, 4, 6)
    cube_owner = 0
    match cube_owner_mask:
        case 0:
            cube_owner = 0
        case 1:
            cube_owner = 1
        case 3:
            cube_owner = None
        case _:
            assert False

    player_on_roll = bit(bs, 6)
    crawford = True if bit(bs, 7) else False

    gamestate = GameState.NO_GAME_STARTED
    match n(bs, 8, 11):
        case 0:
            gamestate = GameState.NO_GAME_STARTED
        case 1:
            gamestate = GameState.PLAYING
        case 2:
            gamestate = GameState.GAME_OVER
        case 3:
            gamestate = GameState.RESIGNED
        case 4:
            gamestate = GameState.OVER_BY_DROP
        case _:
            assert False

    # bit 12
    turn = bit(bs, 11)
    # bit 13
    double = bit(bs, 12)

    # Bit 14-15 indicates whether an resignation was offered.
    #   00 for no resignation,
    #   01 for resign of a single game,
    #   10 for resign of a gammon,
    #   or 11 for resign of a backgammon.
    # The player offering the resignation is the inverse of bit 12, e.g.,
    # if player 0 resigns a gammon then bit 12 will be 1
    # (as it is now player 1 now has to decide whether to
    # accept or reject the resignation) and bit 13-14 will be
    # 10 for resign of a gammon.
    res_flag = n(bs, 13, 15)
    match res_flag:
        case 0:
            resign_flag = ResignFlag.NO_RESIGNATION
        case 1:
            resign_flag = ResignFlag.SINGLE_GAME_OFFERED
        case 2:
            resign_flag = ResignFlag.GAMMON_OFFERED
        case 3:
            resign_flag = ResignFlag.BACKGAMMON_OFFERED
        case _:
            assert False

    d1 = n(bs, 15, 18)
    d2 = n(bs, 18, 21)
    dice = (d1, d2)
    match_length = n(bs, 21, 36)
    player_1_score = n(bs, 36, 51)
    player_2_score = n(bs, 51, 66)
    score = (player_1_score, player_2_score)

    return Match(
        cube=cube,
        cube_owner=cube_owner,
        dice_owner=player_on_roll,
        crawford=crawford,
        gamestate=gamestate,
        turn=turn,
        double=double,
        resign_flag=resign_flag,
        dice=dice,
        match_length=match_length,
        score=score,
    )


def encode_position(board):
    bs = [0 for _ in range(10)]
    i = 0
    mask = 0

    for x in board[1:]:
        while x > 0:
            bs[i] |= 1 << mask
            x -= 1
            mask += 1
            if mask == 8:
                mask = 0
                i += 1
        mask += 1
        if mask == 8:
            mask = 0
            i += 1
    for x in reversed(board[0:25]):
        while x < 0:
            bs[i] |= 1 << mask
            x += 1
            mask += 1
            if mask == 8:
                mask = 0
                i += 1
        mask += 1
        if mask == 8:
            mask = 0
            i += 1
    s = str(base64.b64encode(bytes(bs)))
    return s[: s.find("=")]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("position")
    args = parser.parse_args()
    position = args.position
    try:
        x = decode_position(position)
        print(x)
        y = encode_position(x)
        print(y)
    except:
        x = decode_match(position)
        print(x)
