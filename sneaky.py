
from collections import namedtuple
from typing import NamedTuple

class Node[T](NamedTuple):
    head: T
    tail: 'Node[T] | None'

Move = namedtuple('Move', ['source', 'destination'])
Pip = namedtuple('Pip', ['number', 'checker_count'])

def canonically_sorted_moves(moves):
    return [x for x in reversed(sorted([[x for x in reversed(sorted([Move(src, dest) for (src, dest) in move]))] for move in moves], key=tuple))]    

def from_list(xs, tail=None):
    for x in reversed(xs):
        tail = Node(head=x, tail=tail)
    return tail

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
    result = set()
    move_combinations = [[d1, d2], [d2, d1]] if d1 != d2 else [[d1, d1, d1, d1]]
    stack = from_list([(board, from_list(dice), None) for dice in move_combinations])
    while stack:
        (board, dice, moves) = stack.head
        stack = stack.tail
        if not dice:
            if moves:
                xs = [x for x in reversed(sorted(to_list(moves)))]
                result.add(from_list(xs))
            continue
        (die, dice) = dice
        while board:
            (number, checker_count) = board.head
            destination = number - die
            move = Move(source=number, destination=destination)
            moves = Node(head=move, tail=moves)
            tuple = (board,  dice, moves)
            stack = Node(tuple, tail=stack)
            board = board.tail
            

    result = [x for x in reversed(sorted([to_list(x) for x in list(result)]))]
    return result
