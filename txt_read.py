import mat_parser


def load_match(f):
    s = f.read()
    (x, remainder) = mat_parser.file.parse_partial(s)
    print(x)
    print("-------")
    print(remainder)


if __name__ == "__main__":
    import sys

    load_match(sys.stdin)
