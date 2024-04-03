import sys
import mat_parser

import argparse


def p(x):
    match x:
        case "move_line":
            return mat_parser.move_line
        case "game":
            return mat_parser.game
        case "match":
            return mat_parser.file
        case _:
            raise Exception("unknown parser " + x)


parser = argparse.ArgumentParser()
parser.add_argument("parser", type=p)
args = parser.parse_args()
s = sys.stdin.read()
result = args.parser.parse(s)
print(result)
