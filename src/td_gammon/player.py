from logging import INFO, basicConfig, getLogger
import sys

import regex

import torch

from . import gnubg_codec

from . import backgammon_env

basicConfig(format="%(message)s")
logger = getLogger(__name__)
logger.setLevel(INFO)

position_regex = regex.compile(r"^\s*GNU Backgammon  Position ID: (\S*)\s*$")
match_regex = regex.compile(r"^\s*Match ID   : (\S*)\s*$")

from . import agent


def decide_action(agent: agent.Agent, state, dice):
    state_old = state
    (_, board_next) = agent.next(state, dice)

    bbb = [int(x) for x in board_next.tolist()[0][:26]]
    bck = backgammon_env.Backgammon()
    x = state_old.tolist()[0]
    s = (x[0:26], bool(x[26]), dice)
    for m in bck.available_moves(s):
        (board, _, _) = bck.next(s, m)
        if board == bbb:
            return m
    assert False


def response(position_id, match_id, agent):
    position = gnubg_codec.decode_position(position_id)
    m = gnubg_codec.decode_match(match_id)
    logger.info("decoded position", position)
    logger.info(m)
    if m.gamestate != gnubg_codec.GameState.PLAYING:
        logger.info("gamestate %s", m.gamestate)
        return []
    if m.turn == 1:
        if m.double_being_offered:
            logger.info("taking double")
            return ["take"]
        if m.resign_flag != gnubg_codec.ResignFlag.NO_RESIGNATION:
            logger.info("declining resignation")
            return ["decline"]
        (d1, d2) = m.dice
        logger.info("dice %d %d", d1, d2)
        if d1 == 0:
            logger.info("we are rolling")
            return ["roll"]
        assert d2 != 0
        # its always player 2?
        player_1 = False

        decision = decide_action(
            agent, torch.tensor([position + [1 if player_1 else 0]]), m.dice
        )
        if decision:
            tokens = ["move"]
            for s, e in decision:
                tokens.append(str(s) if s < 25 else "b")
                tokens.append(str(e) if e > 0 else "o")
            x = [" ".join(tokens)]
            logger.info("decision %s", x)
            return x
        else:
            return []
    else:
        logger.info("not our turn")
        return []


def play(agent, n):
    print("new session {n}".format(n=n))

    state = (None, None)
    next_position = None
    next_match = None
    for line in sys.stdin:
        logger.info(line.rstrip())
        if line.startswith("Session complete"):
            print("exit")
            sys.exit(0)
        if m := position_regex.match(line):
            position_id = m.group(1)
            next_position = position_id
        elif m := match_regex.match(line):
            next_match = m.group(1)
            if state != (next_position, next_match):
                state = (next_position, next_match)
                r = response(*state, agent)
                for line in r:
                    print(line)
