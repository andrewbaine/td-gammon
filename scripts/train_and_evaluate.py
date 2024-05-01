import argparse
import logging
import os
import subprocess
import time

logging.basicConfig(level=logging.INFO)


def use_docker_to_check_cuda():
    cp = subprocess.run(["docker", "run", "--rm", "--gpus", "all", "hello-world"])
    logging.info("checking docker returned %d", cp.returncode)
    return cp.returncode == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("-i", "--iterations", type=int, default=100)
    parser.add_argument("-f", "--fork", type=str)
    parser.add_argument("-c", "--continue", type=str)
    parser.add_argument("-d", "--epc-db", type=str, required=False)
    parser.add_argument("--hidden", type=int, required=False)
    parser.add_argument("-o", "--outputs", type=int, required=False)
    parser.add_argument("-a", "--alpha", type=float, default=0.06, dest="α")
    parser.add_argument("-l", "--lambda", type=float, default=0.1, dest="λ")
    parser.add_argument("-e", "--encoding", choices=["baine", "baine_epc", "tesauro"])
    parser.add_argument("--force-cuda", action="store_true")
    args = parser.parse_args()

    use_cuda = args.force_cuda or use_docker_to_check_cuda()

    i = 0

    model = "{encoding}-{hidden}-{out}-{games}-{t}".format(
        encoding=args.encoding,
        hidden=args.hidden,
        out=args.outputs,
        games=args.iterations,
        t=int(time.time()),
    )

    command = ["docker", "run", "--rm"]
    if use_cuda:
        command = command + ["--gpus", "all"]

    command = command + [
        "--mount",
        "type=bind,src={pwd}/var/models,target=/var/models".format(pwd=os.getcwd()),
    ]
    if args.epc_db is not None:
        command = command + [
            "--mount",
            "type=bind,src={pwd}/{epc_db},target=/var/epc_db".format(
                pwd=os.getcwd(), epc_db=args.epc_db
            ),
        ]
    command += ["td-gammon", "train"]
    if use_cuda:
        command += ["--force-cuda"]
    command += ["--alpha", str(args.α)]
    command += ["--lambda", str(args.λ)]

    command += ["--save-dir", "/var/models/{model}".format(model=model)]
    command += ["--out", str(args.outputs)]
    command += ["--encoding", args.encoding]
    command += ["--hidden", str(args.hidden)]
    if args.epc_db is not None:
        command += ["--epc-db", "/var/epc_db"]
    print(command)
    iterations = i + args.step
    command += ["--iterations", "{n}".format(n=0)]

    cp = subprocess.run(command)
    logging.info(cp)
    logging.info("\n".join(command))

    while i < args.iterations:
        command = ["docker", "run", "--rm"]
        if use_cuda:
            command = command + ["--gpus", "all"]

        command = command + [
            "--mount",
            "type=bind,src={pwd}/var/models,target=/var/models".format(pwd=os.getcwd()),
        ]
        if args.epc_db is not None:
            command = command + [
                "--mount",
                "type=bind,src={pwd}/{epc_db},target=/var/epc_db".format(
                    pwd=os.getcwd(), epc_db=args.epc_db
                ),
            ]
        command += ["td-gammon", "train", "--continue"]
        if use_cuda:
            command += ["--force-cuda"]
        command += ["--save-dir", "/var/models/{model}".format(model=model)]
        if args.epc_db is not None:
            command += ["--epc-db", "/var/epc_db"]
        command += ["--iterations", "{n}".format(n=args.step)]
        i += args.step
