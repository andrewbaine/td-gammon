
from collections import namedtuple
from typing import NamedTuple

class Node[T](NamedTuple):
    head: T
    tail: 'Node[T] | None'

Move = namedtuple('Move', ['source', 'destination'])
Pip = namedtuple('Pip', ['number', 'checker_count'])

def from_list(xs, node=None):
    for x in reversed(xs):
        node = Node(head=x, tail=node)
    return node

def to_list(node):
    xs = []
    while node:
        (head, node) = node
        xs.append(head)
    return xs

def make_board():
    tuples = [Pip(24, 2), Pip(19, -5), Pip(17, -3), Pip(13, 5), Pip(12, -5), Pip(8, 3), Pip(6, 5), Pip(1, -2)]
    return from_list(tuples)

def from_board(board):
    return from_list(
        [x for x in reversed([Pip(i, count)  for (i, count) in enumerate(board) if count != 0 and i != 0])])

def moves(board, d1, d2):
    greatest_usage = 0
    result = set()
    move_combinations = [[d1, d2], [d2, d1]] if d1 != d2 else [[d1, d1, d1, d1]]
    stack = [(board, from_list(dice), None, True, False) for dice in move_combinations]
    while stack:
        (board, dice, moves_thus_far, bearoff_allowed, checker_on_higher_number_point) = stack.pop()

        if not dice:
            if moves_thus_far:
                xs = [x for x in reversed(sorted(to_list(moves_thus_far)))]
                greatest_usage = max(greatest_usage, sum([abs(x.source - x.destination) for x in xs]))
                result.add(from_list(xs))
        else:
            (die, dice) = dice
            valid_move_found = False
            while board:
                (source, count) = board.head
                if count > 0:
                    destination = source - die
                    if destination < 1 and not bearoff_allowed:
                        break
                    if destination < 0 and checker_on_higher_number_point:
                        break
                    hs = [Pip(source, count - 1)]
                    t = board.tail
                    while t and destination < t.head.number:
                        (h, t) = t
                        if dice:
                            hs.append(h)
                    if t: # there's more board to iterate over
                        (p, c) = t.head
                        if p == destination and c < -1:
                            # we cant land here, it's occupied by opponent
                            pass
                        else:
                            # we can land here
                            valid_move_found = True
                            moves = Node(Move(source=source, destination=destination), moves_thus_far)
                            if destination == p:
                                if c == -1:
                                    hs.append(Pip(p, 1))
                                elif c > -1:
                                    # we can move here by adding 1
                                    hs.append(Pip(p, c + 1))
                                else:
                                    # we caught this above
                                    assert False
                            elif destination > p:
                                # we can move here, its vacant
                                hs.append(Pip(destination, 1))
                            else:
                                assert False
                            stack.append((from_list(hs, node=t), dice, moves, bearoff_allowed and (count == 1 or source < 7), (checker_on_higher_number_point)))
                    else:
                        # we are landing past the end
                        if destination > 0:
                            hs.append(Pip(destination, 1))
                        moves = Node(Move(source=source, destination=destination), moves_thus_far)
                        tuple = (from_list(hs), dice, moves, bearoff_allowed and (count == 1 or source < 7), checker_on_higher_number_point)
                        stack.append(tuple)
                    if source == 25:
                        break
                bearoff_allowed = bearoff_allowed and (count <= 0 or source < 7)
                checker_on_higher_number_point = checker_on_higher_number_point or (count > 0)
                board = board.tail
            if not valid_move_found:
                stack.append((board, dice, moves_thus_far, bearoff_allowed, checker_on_higher_number_point))

    result = [x for x in reversed(sorted([to_list(x) for x in list(result)])) if sum([abs(x.source - x.destination) for x in x]) == greatest_usage]
    return result
