def play(input):
    lines = []
    bs = []
    while True:
        b = input.read(1)
        if b == "":
            return
        if b == "\n":
            line = "".join(bs)
            lines.append(line)
            bs.clear()
        else:
            bs.append(b)


import sys

from logging import getLogger, INFO, basicConfig

if __name__ == "__main__":
    basicConfig(format="%(message)s")
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    print("new session 1", flush=True)
    for line in sys.stdin:
        #        print("roll")
        logger.info(line.rstrip())
