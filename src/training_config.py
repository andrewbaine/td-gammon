from collections import namedtuple

Config = namedtuple(
    "Config", ["encoding", "hidden", "out", "α", "λ", "iterations", "parent"]
)


def validated_config(encoding, hidden, out, α, λ, iterations, parent):
    assert encoding == "baine" or encoding == "tesauro"
    assert hidden > 0
    assert encoding
    assert hidden
    assert out == 6 or out == 4
    assert 0.0 < α <= 1.0
    assert 0.0 <= λ <= 1.0
    assert iterations > 0
    return validated_config(
        iterations=iterations,
        encoding=encoding,
        hidden=hidden,
        out=out,
        α=α,
        λ=λ,
        parent=parent,
    )


def from_args(args):
    return validated_config(
        iterations=args.iterations,
        encoding=args.encoding,
        hidden=args.hidden,
        out=args.out,
        α=args.α,
        λ=args.λ,
        parent=args.fork,
    )


def from_parent(config, args):
    return validated_config(
        encoding=config.encoding,
        hidden=config.hidden,
        out=config.out,
        iterations=args.iterations,
        α=args.α,
        λ=args.λ,
        parent=args.fork,
    )


def load(path):

    with open(path, "r") as input:
        for line in input:
            tokens = [x.strip() for x in line.split("=")]
            assert len(tokens) == 2
            [key, value] = tokens
            match key:
                case "encoding":
                    encoding = value
                case "hidden":
                    hidden = int(value)
                case "out":
                    out = int(value)
                case "alpha":
                    α = float(value)
                case "lambda":
                    λ = float(value)
                case "iterations":
                    iterations = int(value)
                case "parent":
                    parent = value
                case "move-tensors":
                    pass
                case _:
                    raise Exception("unknown key " + key)
    assert encoding
    assert hidden
    assert out == 6 or out == 4
    assert 0.0 < α < 1.0
    assert 0.0 <= λ <= 1.0
    assert iterations
    return validated_config(
        encoding=encoding,
        hidden=hidden,
        out=out,
        α=α,
        λ=λ,
        iterations=iterations,
        parent=parent,
    )


def store(config, path):
    with open(path, "w") as out:
        for line in [
            "iterations={n}".format(n=config.iterations),
            "encoding={e}".format(e=config.encoding),
            "hidden={n}".format(n=config.hidden),
            "out={n}".format(n=config.out),
            "alpha={x:.8f}".format(x=config.α),
            "lambda={x:.8f}".format(x=config.λ),
            "parent={p}".format(p=config.parent),
        ]:
            out.write(line + "\n")
