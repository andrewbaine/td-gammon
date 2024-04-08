import socket
import sys
import subprocess
import threading


def gnubg(player, port):
    input = "".join(
        [
            x + "\n"
            for x in [
                "set player {player} external localhost:{port}".format(
                    player=player, port=port
                ),
                "new session 1",
            ]
        ]
    )
    subprocess.run(["gnubg", "-q", "-t"], input=input, text=True)


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


def listen(server):
    (connection, x) = server.accept()
    print("x", x)
    for line in lines(connection):
        #        print(line)
        response = sys.stdin.readline()
        connection.send(response.encode())


def main(player):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_address = ("localhost", 0)
    server.bind(server_address)
    server.listen(1)

    s = server.getsockname()
    port = s[1]
    print(input)
    t1 = threading.Thread(target=gnubg, args=(player, port))
    t2 = threading.Thread(target=listen, args=(server,))
    t1.start()
    t2.start()
    t1.join()
    sys.exit(0)


#    server.close()


if __name__ == "__main__":
    main("andrewbaine")
