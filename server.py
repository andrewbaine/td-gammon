import subprocess
import argparse
from collections import namedtuple
import socket
import policy
import threading

import torch

import backgammon
import backgammon_env
import network
import tesauro

Board = namedtuple(
    "Board",
    [
        "names",
        "match_score",
        "board",
        "turn",
        "dice",
        "cube",
        "may_double",
        "was_doubled",
        "color",
        "direction",
        "home_and_bar",
        "on_home",
        "on_bar",
        "can_move",
        "forced_move",
        "did_crawford",
        "max_redoubles",
    ],
)


def get_board(line):

    print("processing", line)
    tokens = line.split(":")
    print(tokens)
    print(len(tokens))
    assert len(tokens) == 53

    assert tokens[0] == "board"
    names = (tokens[1], tokens[2])
    match_length = int(tokens[3])
    my_score = int(tokens[4])
    opponents_score = int(tokens[5])

    return Board(
        names=names,
        match_score=(match_length, my_score, opponents_score),
        board=[int(t) for t in tokens[6:32]],
        turn=int(tokens[32]),
        dice=(((int(tokens[33]), int(tokens[34])), (int(tokens[35]), int(tokens[36])))),
        cube=int(tokens[37]),
        may_double=(int(tokens[38]), int(tokens[39])),
        was_doubled=int(tokens[40]),
        color=int(tokens[41]),
        direction=int(tokens[42]),
        home_and_bar=(int(tokens[43]), int(tokens[44])),
        on_home=(int(tokens[45]), int(tokens[46])),
        on_bar=(int(tokens[47]), int(tokens[48])),
        can_move=int(tokens[49]),
        forced_move=int(tokens[50]),
        did_crawford=int(tokens[51]),
        max_redoubles=int(tokens[52]),
    )


buffer_size = 4096


def lines(connection):
    reader = connection.makefile(mode="rb")
    chunks = []
    while True:
        chunk = reader.read(1)
        if chunk:
            if chunk[0] != 0:
                chunks.append(chunk)
            else:
                word = b"".join(chunks).decode().strip()
                chunks.clear()
                yield word
        else:
            return


def get_response(pol, b):
    player_1 = b.turn == 1
    (dice, dice_1) = b.dice
    assert dice == dice_1

    if dice == (0, 0):
        if b.was_doubled:
            return "take"
        else:
            return "roll"
    else:
        move = pol.choose_action((b.board, player_1, dice))
        tokens = []
        if move:
            for src, dest in move:
                tokens.append("b" if src == 25 else str(src))
                tokens.append("o" if dest <= 0 else str(dest))
        s = " ".join(tokens)
        return s


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=40)
    parser.add_argument("--model", required=True)
    parser.add_argument("--encoding", choices=["tesauro198"], required=True)
    parser.add_argument(
        "--softmax", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction)
    parser.add_argument("--port", type=int, default=8901)

    args = parser.parse_args()
    bck = backgammon_env.Backgammon()
    observe = None

    match args.encoding:
        case "tesauro198":
            observe = tesauro.observe
        case _:
            assert False
    assert observe is not None

    t = observe(bck.s0(player_1=True))
    layers = [t.size()[0], args.hidden, 4]

    nn = network.layered(*layers, softmax=args.softmax)
    nn.load_state_dict(torch.load(args.model))
    nn = network.with_utility(nn)
    pol = policy.Policy_1_ply(bck, observe, nn)
    t = threading.Thread(target=serve, args=(args.port, pol, args.debug))
    t.start()

    input = "\n".join(
        [
            "new session",
            "set player andrewbaine external localhost:8901",
            "set jacoby off",
            "new session 2",
            "export session text session.txt",
            "",
        ]
    )
    completed_process = subprocess.run(
        ["gnubg", "-q", "-t"], input=input, capture_output=True, text=True
    )
    t.join()
    print(completed_process)
    print("done")
    connection = None


def serve(port, pol, debug=False):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_address = ("localhost", port)
    server.bind(server_address)

    server.listen(1)

    try:
        connection, _ = server.accept()
        try:
            with torch.no_grad():
                for line in lines(connection):
                    b = get_board(line)
                    response = get_response(pol, b)
                    if debug:
                        print(line)
                        print(backgammon.to_str(b.board))
                        print("dice", b.dice[0])
                        print("response", response)
                    connection.send((response + "\n").encode())
        except KeyboardInterrupt:
            if connection:
                connection.close()
        finally:
            connection.close()
    finally:
        server.close()


if __name__ == "__main__":
    main()
