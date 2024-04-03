from collections import namedtuple

Action = namedtuple("Action", ["name", "details"])

from parsy import decimal_digit, generate, regex, string, whitespace

number = regex(r"\d+").map(int)

end_of_line = regex(r"[ \t]*\n")


@generate
def optional_whitespace():
    yield regex(r"[ \t]*")


@generate
def required_whitespace():
    yield regex(r"[ \t]+")


newline = string("\n")
blank_line = regex(r"[ \t]*\n")


@generate("player")
def player():
    tokens = yield regex(r"[^:\s]*").sep_by(required_whitespace)
    yield optional_whitespace
    yield string(":")
    yield optional_whitespace
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
    yield optional_whitespace
    yield string("=>")
    yield optional_whitespace
    n = yield number
    return Action(name="double", details=n)


@generate("dice")
def dice():
    yield optional_whitespace
    d1 = yield decimal_digit.map(int)
    d2 = yield decimal_digit.map(int)
    return (d1, d2)


pip = number
bar = string("bar")
src = pip | bar
off = string("off")
dest = pip | off


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

many_moves = move_checker.sep_by(optional_whitespace)

decision = cannot_move | many_moves | string("")

Play = namedtuple("Play", ["dice", "actions"])


@generate("play")
def play():
    d = yield dice
    yield string(":")
    yield optional_whitespace
    actions = yield decision
    return Play(dice=d, actions=actions)


@generate("move_number")
def move_number():
    n = yield number
    yield string(")")
    return n


@generate("summary")
def summary():
    s = yield regex(r"Wins \d+ point(s?)")
    return s


turn = (
    double
    | string("Takes").map(lambda x: Action(name="take", details=None))
    | string("Drops").map(lambda x: Action(name="drop", details=None))
    | play.map(lambda x: Action(name="play", details=x))
    | summary.map(lambda x: Action(name="summary", details=x))
)


@generate
def comment_line():
    yield string(";")
    yield optional_whitespace
    yield string("[")
    yield optional_whitespace
    key = yield regex(r'([^"]*)')
    yield string('"')
    value = yield regex(r'([^"]*)')
    yield string('"')
    yield optional_whitespace
    yield string("]")
    yield end_of_line
    return (key, value)


@generate("summary_line")
def summary_line():
    yield optional_whitespace
    s = yield summary
    suffix = yield string(" and the match").optional(default="")
    yield end_of_line
    return s + suffix


@generate("move_line")
def move_line():
    yield optional_whitespace
    n = yield move_number
    yield optional_whitespace
    turns = yield turn.sep_by(optional_whitespace, min=0, max=2)
    yield end_of_line
    return (n, turns)


@generate("game_number_line")
def game_number_line():
    yield optional_whitespace
    n = yield regex(r"Game (\d+)", group=1).map(int)
    yield end_of_line


@generate("player_line")
def player_line():
    yield optional_whitespace
    p1 = yield player
    yield optional_whitespace
    p2 = yield player
    yield end_of_line
    return (p1, p2)


@generate("match_length_line")
def match_length_line():
    yield optional_whitespace
    match_length = yield regex(r"(\d+) point match", group=1).map(int)
    yield end_of_line
    return match_length


@generate("game")
def game():
    yield blank_line.many()
    game_number = yield game_number_line
    players = yield player_line
    yield blank_line.many()
    moves = yield move_line.many()
    sum = yield summary_line.optional()
    return (game_number, players, moves, sum)


@generate
def file():
    comments = yield comment_line.many()
    yield blank_line.many()
    m = yield match_length_line
    games = yield game.many()
    return (comments, m, games)
