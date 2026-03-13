from collections.abc import Callable
from collections import deque
from game import *
from .player_board import PlayerBoard


POWERUP_THRESHOLD   = 70
DOMINANCE_RATIO     = 0.60
BID_AMOUNT          = 40
MEMORY              = 6
BACKTRACK_PENALTY   = 40.0


class PlayerController:

    def __init__(self, player_parity: int, time_left: Callable):
        self.last_loc = None
        self.history: deque = deque(maxlen=MEMORY)
        self.turn = 0

    def bid(self, board: Board, player_parity: int, time_left: Callable) -> int:
        return BID_AMOUNT

    def play(self, board: Board, player_parity: int, time_left: Callable):
        self.turn += 1
        pb = PlayerBoard(board, player_parity)
        player = pb.get_player()
        loc, stamina = player.loc, player.stamina

        all_moves = pb.get_valid_non_beacon_moves()
        if not all_moves:
            return []

        opp_reachable = self._opp_reachable(pb)

        # Partition moves into safe and unsafe ONCE, clearly
        safe_moves = [m for m in all_moves if self._is_safe(pb, loc, m, opp_reachable)]
        # Prefer non-backtrack, but never leave empty
        safe_nb = [m for m in safe_moves if self._dest(loc, m) != self.last_loc]
        working_moves = safe_nb if safe_nb else (safe_moves if safe_moves else all_moves)

        actions = []
        m1 = self._pick(pb, working_moves, opp_reachable, loc, stamina)
        actions.append(m1)

        pb1 = pb.get_copy()
        pb1.apply_action(m1, moves_this_turn=0)
        p1 = pb1.get_player()
        loc1, stamina1 = p1.loc, p1.stamina
        if pb1.board.cells[loc1.r][loc1.c].powerup:
            stamina1 = min(stamina1 + GameConstants.STAMINA_POWERUP_AMOUNT, p1.max_stamina)

        paints, stamina1 = self._paint_adj(pb1, loc1, stamina1)
        actions.extend(paints)

        self.history.append(loc)
        self.last_loc = loc

        cost2 = GameConstants.EXTRA_MOVE_COST
        if stamina1 >= cost2 + GameConstants.PAINT_STAMINA_COST:
            all_moves2 = pb1.get_valid_non_beacon_moves(moves_this_turn=1)
            opp_reachable2 = self._opp_reachable(pb1)
            safe2 = [m for m in all_moves2 if self._is_safe(pb1, loc1, m, opp_reachable2)]
            safe2_nb = [m for m in safe2 if self._dest(loc1, m) != loc]
            working2 = safe2_nb if safe2_nb else (safe2 if safe2 else all_moves2)
            if working2:
                m2 = self._pick(pb1, working2, opp_reachable2, loc1, stamina1)
                actions.append(m2)
                pb2 = pb1.get_copy()
                pb2.apply_action(m2, moves_this_turn=1)
                p2 = pb2.get_player()
                loc2, stamina2 = p2.loc, p2.stamina - cost2
                if pb2.board.cells[loc2.r][loc2.c].powerup:
                    stamina2 = min(stamina2 + GameConstants.STAMINA_POWERUP_AMOUNT, p2.max_stamina)
                paints2, _ = self._paint_adj(pb2, loc2, stamina2)
                actions.extend(paints2)

        return actions

    # ── Safety ────────────────────────────────────────────────────────────────

    def _dest(self, loc: Location, m) -> Location:
        dr, dc = m.direction.value
        return Location(loc.r + dr, loc.c + dc)

    def _we_own(self, pb: PlayerBoard, loc: Location) -> bool:
        """Cell has our paint on it. paint_value==0 means nobody owns it."""
        cell = pb.board.cells[loc.r][loc.c]
        return cell.paint_value != 0 and Parity.owned(cell.paint_value, pb.player_parity)

    def _they_own(self, pb: PlayerBoard, loc: Location) -> bool:
        cell = pb.board.cells[loc.r][loc.c]
        return cell.paint_value != 0 and Parity.owned(cell.paint_value, pb.opponent_parity)

    def _is_safe(self, pb: PlayerBoard, loc: Location, m, opp_reachable: set) -> bool:
        """
        A move is safe if we cannot lose a collision on the destination.
        We WIN collisions on cells WE own.
        We LOSE collisions on empty cells OR opponent cells if opp can reach them.
        """
        nl = self._dest(loc, m)
        if self._we_own(pb, nl):
            return True   # our paint = we always win collision
        # Empty or opponent cell: safe only if opponent cannot reach it
        return nl not in opp_reachable

    def _opp_reachable(self, pb: PlayerBoard) -> set:
        """All cells opponent can reach this turn given their stamina."""
        opp = pb.get_opponent()
        opp_loc = opp.loc
        opp_stamina = opp.stamina

        best = {opp_loc: opp_stamina}
        queue = deque([(opp_loc, opp_stamina, 0)])
        reachable = set()

        while queue:
            cur, stamina_here, moves_used = queue.popleft()
            if best.get(cur, -1) > stamina_here:
                continue
            # Cost to step away from cur: move 1 is free (moves_used=0), move 2 costs 10, etc.
            next_cost = GameConstants.EXTRA_MOVE_COST * moves_used
            if stamina_here < next_cost:
                continue
            stamina_after = stamina_here - next_cost
            for nb in cur.neighbors():
                if pb.board.oob(nb) or pb.board.cells[nb.r][nb.c].is_wall:
                    continue
                if stamina_after > best.get(nb, -1):
                    best[nb] = stamina_after
                    reachable.add(nb)
                    queue.append((nb, stamina_after, moves_used + 1))

        return reachable

    # ── Strategy ──────────────────────────────────────────────────────────────

    def _pick(self, pb: PlayerBoard, moves, opp_reachable: set, loc: Location, stamina: int):
        hills = self._hills(pb)
        total = len(hills)
        their_count = sum(1 for h in hills.values() if h['they_control'])

        # 1. Kill: step onto our cell or empty cell the opponent can also reach
        for m in moves:
            nl = self._dest(loc, m)
            if nl not in opp_reachable:
                continue
            if self._we_own(pb, nl) or not self._they_own(pb, nl):
                # our cell = guaranteed win; empty cell = initiator wins = us
                # but only if it's truly empty (paint_value==0) or ours
                cell = pb.board.cells[nl.r][nl.c]
                if self._we_own(pb, nl) or cell.paint_value == 0:
                    return m

        # 2. Dominance guard
        if total > 0 and their_count / total >= DOMINANCE_RATIO:
            targets = {l for h in hills.values() if h['they_control']
                       for l in h['cells'] if self._they_own(pb, l)}
            if targets:
                m = self._bfs(pb, loc, moves, lambda n: n in targets)
                if m: return m

        # 3. Nearest unpainted hill cell
        unpainted_hill = {
            l for h in hills.values() for l in h['cells']
            if not self._we_own(pb, l)
        }
        if unpainted_hill:
            m = self._bfs(pb, loc, moves, lambda n: n in unpainted_hill)
            if m: return m

        # 4. Powerup
        if stamina < POWERUP_THRESHOLD:
            m = self._bfs(pb, loc, moves, lambda n: bool(pb.board.cells[n.r][n.c].powerup))
            if m: return m

        # 5. Expand to unpainted cells
        m = self._bfs(pb, loc, moves, lambda n: pb.board.cells[n.r][n.c].paint_value == 0)
        if m: return m

        # 6. Roam
        return self._roam(pb, loc, moves)

    # ── Hill analysis ─────────────────────────────────────────────────────────

    def _hills(self, pb: PlayerBoard) -> dict:
        hills = {}
        rows, cols = pb.board.board_size.r, pb.board.board_size.c
        thresh = GameConstants.HILL_CONTROL_THRESHOLD
        for r in range(rows):
            for c in range(cols):
                cell = pb.board.cells[r][c]
                hid = cell.hill_id
                if not hid:
                    continue
                if hid not in hills:
                    hills[hid] = {'cells': [], 'ours': 0, 'theirs': 0}
                l = Location(r, c)
                hills[hid]['cells'].append(l)
                if self._we_own(pb, l):
                    hills[hid]['ours'] += 1
                elif self._they_own(pb, l):
                    hills[hid]['theirs'] += 1
        for h in hills.values():
            t = len(h['cells'])
            h['we_control'] = h['ours'] > h['theirs'] and h['ours'] / t >= thresh
            h['they_control'] = h['theirs'] > h['ours'] and h['theirs'] / t >= thresh
        return hills

    # ── BFS ───────────────────────────────────────────────────────────────────

    def _bfs(self, pb: PlayerBoard, start: Location, moves, pred):
        move_map = {}
        for m in moves:
            nl = self._dest(start, m)
            if not pb.board.oob(nl) and not pb.board.cells[nl.r][nl.c].is_wall:
                move_map.setdefault(nl, m)

        visited = {start}
        queue = deque()
        for nl, m in move_map.items():
            if pred(nl): return m
            visited.add(nl)
            queue.append((nl, m))

        while queue:
            cur, fm = queue.popleft()
            for n in cur.neighbors():
                if n in visited or pb.board.oob(n) or pb.board.cells[n.r][n.c].is_wall:
                    continue
                visited.add(n)
                if pred(n): return fm
                queue.append((n, fm))
        return None

    # ── Roam ──────────────────────────────────────────────────────────────────

    def _roam(self, pb: PlayerBoard, loc: Location, moves):
        move_map = {}
        for m in moves:
            nl = self._dest(loc, m)
            if not pb.board.oob(nl) and not pb.board.cells[nl.r][nl.c].is_wall:
                move_map.setdefault(nl, m)
        if not move_map: return None

        visited = {loc}
        queue = deque()
        cell_data = {}
        for nl, m in move_map.items():
            visited.add(nl)
            cell_data[nl] = (m, 1)
            queue.append((nl, m, 1))

        while queue:
            cur, fm, d = queue.popleft()
            for n in cur.neighbors():
                if n in visited or pb.board.oob(n) or pb.board.cells[n.r][n.c].is_wall:
                    continue
                visited.add(n)
                cell_data[n] = (fm, d + 1)
                queue.append((n, fm, d + 1))

        recent = set(self.history)
        best_score, best_move = -float('inf'), None
        for nl, (fm, d) in cell_data.items():
            s = self._cell_score(pb, nl) - d * 0.5
            if nl in recent: s -= BACKTRACK_PENALTY
            if s > best_score:
                best_score, best_move = s, fm
        return best_move

    def _cell_score(self, pb: PlayerBoard, loc: Location) -> float:
        cell = pb.board.cells[loc.r][loc.c]
        is_hill = bool(cell.hill_id)
        ours = self._we_own(pb, loc)
        theirs = self._they_own(pb, loc)
        unpainted = cell.paint_value == 0

        if is_hill and unpainted:    score = 100
        elif is_hill and theirs:     score = 80
        elif is_hill and not ours:   score = 60
        elif is_hill:                score = 20
        elif unpainted:              score = 30
        elif theirs:                 score = 10
        else:                        score = 5

        if cell.powerup: score += 50
        if self.last_loc and loc == self.last_loc: score -= 20
        return score

    # ── Painting ──────────────────────────────────────────────────────────────

    def _paint_adj(self, pb: PlayerBoard, loc: Location, stamina: int):
        actions = []
        candidates = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            n = Location(loc.r + dr, loc.c + dc)
            if pb.board.oob(n): continue
            cell = pb.board.cells[n.r][n.c]
            if cell.is_wall or cell.beacon_parity != 0: continue
            if self._they_own(pb, n): continue
            if self._we_own(pb, n) and abs(cell.paint_value) >= GameConstants.MAX_PAINT_VALUE: continue
            pri = (10 if cell.hill_id and cell.paint_value == 0 else
                   8  if cell.hill_id else
                   6  if cell.paint_value == 0 else
                   max(0, 4 - abs(cell.paint_value)))
            candidates.append((pri, len(candidates), n))

        candidates.sort(reverse=True)
        for _, _i, n in candidates:
            if stamina < GameConstants.PAINT_STAMINA_COST: break
            actions.append(Action.Paint(n))
            stamina -= GameConstants.PAINT_STAMINA_COST
            pb.apply_action(Action.Paint(n))
        return actions, stamina

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _no_backtrack(self, moves, loc: Location):
        if self.last_loc is None: return moves
        filtered = [m for m in moves if self._dest(loc, m) != self.last_loc]
        return filtered if filtered else moves

    def commentate(self, board: Board, player_parity: int, time_left: Callable) -> str:
        pb = PlayerBoard(board, player_parity)
        hills = self._hills(pb)
        ours = sum(1 for h in hills.values() if h['we_control'])
        theirs = sum(1 for h in hills.values() if h['they_control'])
        return f"T{self.turn} hills={ours}v{theirs}/{len(hills)} stamina={pb.get_player().stamina}"