from collections import namedtuple
import socket
import subprocess
import subprocess
import threading

import torch

import backgammon


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
    tokens = line.split(":")
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


def count(haystack, needle):
    n = 0
    start = 0
    while True:
        i = haystack.find(needle, start)
        if i == -1:
            break
        else:
            n += 1
            start = i + len(needle)
    return n


def serve(server, pol, debug=False):
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


def try_gnubg(pol, games=1, debug=False):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_address = ("localhost", 0)
    server.bind(server_address)
    s = server.getsockname()
    port = s[1]

    server.listen(1)

    t = threading.Thread(target=serve, args=(server, pol, debug))
    t.start()

    benchmark_player = "gnubg"
    player_under_evaluation = "andrewbaine"

    input = "\n".join(
        [
            "new session",
            "set player {player} external localhost:{port}".format(
                player=player_under_evaluation, port=port
            ),
            "set jacoby off",
            "new session {games}".format(games=games),
            "",
        ]
    )
    completed_process = subprocess.run(
        ["gnubg", "-q", "-t"], input=input, capture_output=True, text=True
    )
    t.join()
    if completed_process.stderr:
        raise Exception(completed_process.stderr)
    return [
        count(completed_process.stdout, x)
        for x in [
            "{p} wins a backgammon".format(p=benchmark_player),
            "{p} wins a gammon".format(p=benchmark_player),
            "{p} wins a single game".format(p=benchmark_player),
            "{p} wins a single game".format(p=player_under_evaluation),
            "{p} wins a gammon".format(p=player_under_evaluation),
            "{p} wins a backgammon".format(p=player_under_evaluation),
        ]
    ]
