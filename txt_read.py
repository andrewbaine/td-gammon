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
blank_line = regex(r"^\s\t*$").then(newline)


@generate
def file():
    comments = yield comment.sep_by(end_of_line)
    return comments


parser = comment


def load_match(s):
    (x, _) = file.parse_partial(file_content)
    print(x)


if __name__ == "__main__":
    import sys

    load_match(sys.stdin)
