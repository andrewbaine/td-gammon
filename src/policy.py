import heapq
import numpy
import torch


def evaluate_action_1_ply(bck, observe, nn, state, action):
    (board, player_1, _) = state
    s = bck.next(state, action)
    t = torch.tensor(observe(s))
    return nn(t).item()


def choose_action_1_ply(bck, observe, nn, state):
    (board, player_1, dice) = state
    moves = bck.available_moves(state)

    best = None
    best_move = None

    for action in moves:
        y = evaluate_action_1_ply(bck, observe, nn, state, action)
        if best is None or ((y > best) if player_1 else (y < best)):
            best = y
            best_move = action
    return best_move


def evaluate_action_2_ply(bck, observe, nn, state, action):
    (board, player_1, dice) = state
    s = bck.next((board, player_1, dice), action)
    if bck.done(s):
        t = observe(s)
        return nn(t).item()

    equity = 0
    count = 0
    (board, player_1, dice) = s
    for d1 in range(1, 7):
        for d2 in range(d1, 7):
            factor = 1 if d1 == d2 else 2
            dice = (d1, d2)
            action = choose_action_1_ply(bck, observe, nn, state)
            e = evaluate_action_1_ply(bck, observe, nn, state, action)
            equity += factor * e
            count += factor
    return equity / count


def choose_action_2_ply(bck, observe, nn, state):
    (board, player_1, dice) = state
    moves = bck.available_moves((board, player_1), dice)

    best = None
    best_move = None

    for action in moves:
        y = evaluate_action_2_ply(bck, observe, nn, state, action)
        if best is None or ((y > best) if player_1 else (y < best)):
            best = y
            best_move = action
    return best_move


class Policy:
    def __init__(self, bck, observe, nn):
        self._bck = bck
        self._observe = observe
        self._nn = nn

    def choose_action(self, state):
        assert False


class Policy_1_ply(Policy):
    def __init__(self, bck, observe, nn):
        super().__init__(bck, observe, nn)

    def choose_action(self, state):
        return choose_action_1_ply(self._bck, self._observe, self._nn, state)


class Policy_2_ply_exhaustive(Policy_1_ply):
    def __init__(self, bck, observe, nn):
        super().__init__(bck, observe, nn)

    def choose_action(self, state):
        return choose_action_1_ply(self._bck, self._observe, self._nn, state)


class Policy_2_ply_selective(Policy_2_ply_exhaustive):
    def __init__(self, bck, observe, nn, min_comparisons=3, max_comparisons=None):
        super(Policy_2_ply_selective, self).__init__(bck, observe, nn)
        assert min_comparisons is None or min_comparisons > -1
        assert max_comparisons is None or max_comparisons > -1
        assert (
            min_comparisons is None
            or max_comparisons is None
            or min_comparisons < max_comparisons
        )
        self._min_comparisons = min_comparisons
        self._max_comparisons = max_comparisons

    def choose_action(self, state):
        (board, player_1, dice) = state
        moves = self._bck.available_moves((board, player_1, dice))
        if not moves:
            return None

        h = []
        # collect the 1 ply evaluations
        for action in moves:

            v = evaluate_action_1_ply(self._bck, self._observe, self._nn, state, action)
            item = (-1 * v if player_1 else v, action)
            heapq.heappush(h, item)

        # compute the standard deviation
        std = numpy.std([v for (v, _) in h])

        # pop off our best 1_ply candidate
        (best_equity_1_ply, best_action) = heapq.heappop(h)
        best_equity = evaluate_action_2_ply(
            self._bck, self._observe, self._nn, state, best_action
        )
        comparison_count = 0

        # for (potentially every, but probably only a few)
        # candidate in the heap:
        while h:
            (equity_1_ply, action) = heapq.heappop(h)
            stop_allowed = (
                self._min_comparisons is None
                or comparison_count >= self._min_comparisons
            )
            comparison_count_exceeded = (
                self._max_comparisons is not None
                and comparison_count > self._max_comparisons
            )
            move_under_consideration_differs_too_much = (
                abs(equity_1_ply - best_equity_1_ply) > std
            )
            if stop_allowed and (
                comparison_count_exceeded or move_under_consideration_differs_too_much
            ):
                break
            else:
                equity = evaluate_action_2_ply(
                    self._bck, self._observe, self._nn, state, action
                )
                if (equity > best_equity) if player_1 else (equity < best_equity):
                    best_equity = equity
                    best_action = action
            comparison_count += 1
        return best_action
