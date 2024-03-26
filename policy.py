import heapq


class Policy:
    def __init__(self, bck, observe, nn):
        self._bck = bck
        self._observe = observe
        self._nn = nn

    def evaluate_action(self, state, action):
        assert False

    def choose_action(self, state):
        (board, player_1, dice) = state
        moves = self._bck.available_moves((board, player_1), dice)

        best = None
        best_move = None

        for action in moves:
            y = self.evaluate_action(state, action)
            if best is None or ((y > best) if player_1 else (y < best)):
                best = y
                best_move = action
        return best_move


class Policy_1_ply(Policy):
    def __init__(self, bck, observe, nn):
        super().__init__(bck, observe, nn)

    def evaluate_action(self, state, action):
        (board, player_1, _) = state
        s = self._bck.next((board, player_1), action)
        t = self._observe(s)
        return self._nn(t).item()


class Policy_2_ply_exhaustive(Policy):
    def __init__(self, bck, observe, nn):
        super().__init__(bck, observe, nn)

    def evaluate_action(self, state, action):
        (board, player_1, _) = state
        s1 = self._bck.next((board, player_1), action)

        if self._bck.done(s1):
            t = self._observe(s1)
            return self._nn(t).item()
        else:
            (b1, p1) = s1
            equity_sum = 0
            equity_count = 0
            for d1 in range(1, 7):
                for d2 in range(d1, 7):
                    # best 1 ply action
                    a = super().choose_action((b1, p1, (d1, d2)))
                    s2 = self._bck.next(s1, a)
                    t = self._observe(s2)
                    equity = self._nn(t).item()
                    factor = 1 if d1 == d2 else 2
                    equity_sum += equity * factor
                    equity_count += factor
            assert equity_sum == 36
            return equity_sum / equity_count


class Policy_2_ply_seletive(Policy_2_ply_exhaustive):
    def __init__(self, bck, observe, nn, min_comparisons=3, max_comparisons=None):
        super().__init__(bck, observe, nn)
        self._min_comparisons = min_comparisons
        self._max_comparisons = max_comparisons

    def choose_action(self, state):
        (board, player_1, dice) = state
        moves = self._bck.available_moves((board, player_1), dice)
        if not moves:
            return None

        h = []
        vs = []
        # collect the 1 ply evaluations
        for action in moves:
            v = super().evaluate_action(state, action)
            vs.append(v)
            item = (-1 * v if player_1 else v, action)
            heapq.heappush(h, item)

        # compute the standard deviation
        vs = [v for (_, v) in moves]
        mean = mean = sum(vs) / len(vs)
        variance = sum([((x - mean) ** 2) for x in vs]) / len(vs)
        std = variance**0.5

        # pop off our best 1_ply candidate
        (best_equity_1_ply, best_action) = heapq.heappop(h)
        best_equity = super().evaluate_action(state, best_action)
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
            if stop_allowed and (
                comparison_count_exceeded
                or (abs(equity_1_ply - best_equity_1_ply) > std)
            ):
                break
            else:
                equity = super().evaluate_action(state, action)
                if (equity > best_equity) if player_1 else (equity < best_equity):
                    print("2 ply superior!")
                    best_equity = equity
                    best_action = action
            comparison_count += 1
        return best_action
