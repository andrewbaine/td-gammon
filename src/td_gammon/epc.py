import struct


def make_key(board):
    end = len(board)
    while end > 0 and not board[end - 1]:
        end -= 1
    return bytes(board[1:end])


def f(db, board):
    key = make_key(board)
    v = db.get(key)
    if v is None:
        return [0.0, 0.0]
    else:
        (v,) = struct.unpack("f", v)
        return [1.0, v]


def lookup(db, board):
    a = [(x if x > 0 else 0) for x in board[1:26]]
    b = [(-x if x < 0 else 0) for x in reversed(board[0:25])]
    return f(db, a) + f(db, b)
