from collections import namedtuple

Config = namedtuple(
    "Config",
    ["encoding", "hidden", "out", "α", "λ", "iterations", "parent"],
    defaults=["baine", 40, 4, 0.05, 0.99, 1, ""],
)


def from_args(args):
    return Config(
        iterations=args.iterations,
        encoding=args.encoding,
        hidden=args.hidden,
        out=args.out,
        α=args.α,
        λ=args.λ,
        parent=args.fork,
    )


def from_parent(config, args):
    return config._replace(
        parent=args.fork, iterations=args.iterations, α=args.α, λ=args.λ
    )


def load(path):
    config = Config()
    with open(path, "r") as input:
        for line in input:
            tokens = [x.strip() for x in line.split("=")]
            assert len(tokens) == 2
            [key, value] = tokens
            match key:
                case "encoding":
                    config = config._replace(encoding=value)
                case "hidden":
                    config = config._replace(hidden=int(value))
                case "out":
                    config = config._replace(out=int(value))
                case "alpha":
                    config = config._replace(α=float(value))
                case "lambda":
                    config = config._replace(λ=float(value))
                case "iterations":
                    config = config._replace(iterations=int(value))
                case "parent":
                    config = config._replace(parent=value)
                case "move-tensors":
                    pass
                case _:
                    raise Exception("unknown key " + key)
    return config


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
