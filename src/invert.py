import backgammon
import sys

if __name__ == "__main__":
    text = sys.stdin.read()
    board = backgammon.from_str(text)
    backgammon.invert(board)
    print(backgammon.to_str(board))
