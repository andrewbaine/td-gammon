import socket
import sys

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_address = ("localhost", 0)
server.bind(server_address)
s = server.getsockname()
port = s[1]
print(port)
server.listen(1)
try:
    connection, _ = server.accept()
    try:
        for line in sys.stdin:
            connection.send(line.encode())
    except KeyboardInterrupt:
        if connection:
            connection.close()
    finally:
        connection.close()
finally:
    server.close()
