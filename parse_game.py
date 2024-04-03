import sys

import mat_parser

s = sys.stdin.read()
game = mat_parser.game.parse(s)
print(game)
