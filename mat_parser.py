from parsy import decimal_digit, generate, regex, string, whitespace

semicolon = string(";")

end_of_line = regex(r"[ \t]*\n")


@generate
def comment():
    yield semicolon
    yield whitespace.optional()
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
    yield whitespace.optional()
    p = yield regex(r"[^:]*").map(lambda s: s.strip())
    yield string(":")
    yield whitespace.optional()
    c = yield regex(r"\d+").map(int)
    return (p, c)


@generate("cannot_move")
def cannot_move():
    yield string("Cannot Move")
    return []


@generate("roll")
def roll():
    yield whitespace.optional()
    d1 = yield decimal_digit.map(int)
    d2 = yield decimal_digit.map(int)
    return (d1, d2)


number = regex(r"\d+").map(int)
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

decision = cannot_move | move_checker.sep_by(whitespace)


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


@generate("move_line")
def move_line():
    yield whitespace
    n = yield move_number
    a1 = yield action
    yield whitespace
    a2 = yield action.optional()
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
