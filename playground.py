import gnubg
from subprocess import Popen, PIPE
import sys
import regex

position_regex = regex.compile(r"^\s*GNU Backgammon  Position ID: (\S*)\s*$")
match_regex = regex.compile(r"^\s*Match ID   : (\S*)\s*$")

with Popen(
    ["/usr/local/bin/gnubg", "-q", "-t"], stdout=PIPE, stdin=PIPE, text=True
) as proc:

    def write(s):
        sys.stdout.write("\tsent to gnubg via stdin: " + s + "\n")
        proc.stdin.write(s + "\n")
        proc.stdin.flush()

    write("set player gnubg human")
    write("new game")

    while True:
        position_id = None
        match_id = None
        while True:
            line = proc.stdout.readline()
            sys.stdout.write(line)
            sys.stdout.flush()

            m = position_regex.match(line)
            if m:
                position_id = m.group(1)
            m = match_regex.match(line)
            if m:
                match_id = m.group(1)

            if position_id and match_id:
                p = gnubg.decode_position(position_id)
                m = gnubg.decode_match(match_id)
                print(p)
                print(m)
                break
