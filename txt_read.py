import mat_parser


def load_match(f):
    s = f.read()
    x = mat_parser.file.parse(s)
    print(x)


if __name__ == "__main__":
    import sys

    load_match(sys.stdin)
