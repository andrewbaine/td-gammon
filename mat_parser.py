from collections import namedtuple

Action = namedtuple("Action", ["name", "details"])

from parsy import decimal_digit, generate, regex, string, whitespace

number = regex(r"\d+").map(int)

end_of_line = regex(r"[ \t]*\n")


@generate
def comment():
    yield string(";")
    yield whitespace
    yield string("[")
    yield whitespace.optional()
    key = yield regex(r'([^"]*)')
    yield string('"')
    value = yield regex(r'([^"]*)')
    yield string('"')
    yield whitespace.optional()
    yield string("]")
    return (key, value)


newline = string("\n")
blank_line = regex(r"[ \t]*\n")

match_length = regex(r"(\d+) point match", group=1).map(int)


@generate("player")
def player():
    tokens = yield regex(r"[^:\s]*").sep_by(whitespace)
    yield whitespace.optional()
    yield string(":")
    yield whitespace.optional()
    c = yield number
    p = " ".join(tokens)
    return (p, c)


@generate("cannot_move")
def cannot_move():
    yield string("Cannot Move")
    return []


@generate("double")
def double():
    yield string("Doubles")
    yield whitespace
    yield string("=>")
    yield whitespace
    n = yield number
    return Action(name="double", details=n)


@generate("roll")
def roll():
    yield whitespace.optional()
    d1 = yield decimal_digit.map(int)
    d2 = yield decimal_digit.map(int)
    return (d1, d2)


pip = number
bar = string("bar")
src = pip | bar
dest = pip


@generate("modifier")
def modifier():
    yield string("(")
    n = yield number
    yield string(")")
    return n


@generate("move_checker")
def move_checker():
    s = yield src
    yield string("/")
    d = yield dest
    hit = yield string("*").optional().map(lambda x: x == "*")
    mod = yield modifier.optional()
    return (s, d, hit, mod)


#    return (s, d, hit)

many_moves = move_checker.sep_by(whitespace)

decision = cannot_move | many_moves


@generate("action")
def action():
    dice = yield roll
    yield string(":")
    yield whitespace
    actions = yield decision
    return (dice, actions)


@generate("move_number")
def move_number():
    n = yield number
    yield string(")")
    return n


@generate("drop")
def drop():
    yield string("Drops")
    return Action(name="drop", details=None)


play_or_double_or_take_or_drop = double | action | drop


@generate("move_line")
def move_line():
    yield whitespace
    n = yield move_number
    yield whitespace
    a1 = yield play_or_double_or_take_or_drop
    yield whitespace
    a2 = yield play_or_double_or_take_or_drop.optional()
    return (n, a1, a2)


@generate("game")
def game():
    game_number = yield regex(r"Game (\d+)", group=1).map(int)
    yield whitespace
    player_1 = yield player
    yield whitespace
    player_2 = yield player
    yield end_of_line
    moves = yield move_line.many()
    return (game_number, player_1, player_2, moves)


@generate
def file():
    comments = yield comment.sep_by(end_of_line)
    yield whitespace
    m = yield match_length
    yield whitespace
    games = yield game.many()
    return (comments, m, games)
