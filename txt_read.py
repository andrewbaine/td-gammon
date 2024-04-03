from parsy import generate, regex, string, whitespace

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


@generate
def file():
    comments = yield comment.sep_by(end_of_line)
    yield whitespace
    m = yield match_length
    yield whitespace
    return comments, m


parser = comment


def load_match(f):
    s = f.read()
    (x, remainder) = file.parse_partial(s)
    print(x)
    print("-------")
    print(remainder)


if __name__ == "__main__":
    import sys

    load_match(sys.stdin)
