import regex
import sys
import gnubg_codec

from logging import getLogger, INFO, basicConfig

basicConfig(format="%(message)s")
logger = getLogger(__name__)
logger.setLevel(INFO)

position_regex = regex.compile(r"^\s*GNU Backgammon  Position ID: (\S*)\s*$")
match_regex = regex.compile(r"^\s*Match ID   : (\S*)\s*$")


def response(position_id, match_id, policy):
    position = gnubg_codec.decode_position(position_id)
    m = gnubg_codec.decode_match(match_id)
    logger.info(position)
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
        state = (position, player_1, m.dice)
        decision = policy(state)
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


import backgammon_env
import random

if __name__ == "__main__":
    print("new session 1", flush=True)

    bck = backgammon_env.Backgammon()

    def policy(state):
        ms = bck.available_moves(state)
        if not ms:
            return None
        i = random.randint(0, len(ms) - 1)
        return ms[i]

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
                r = response(*state, policy)
                for line in r:
                    print(line)

        #                for line in r:
        #                    print(r)
