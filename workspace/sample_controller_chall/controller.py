from collections.abc import Callable
from collections import deque
from game import *
from .player_board import PlayerBoard

_W = {
    'score_hill_unpainted': 111.22101979219957,
    'score_hill_theirs': 30.57010408390029,
    'score_hill_not_ours': 42.27767072610716,
    'score_hill_ours': 16.081509610278804,
    'score_unpainted': 25.304111541385844,
    'score_theirs': 6.5414734405451975,
    'score_ours': 3.452518349284261,
    'score_powerup_bonus': 36.842385113159104,
    'score_last_loc_penalty': -51.99831823238514,
    'paint_pri_hill_unpainted': 14.286005831697059,
    'paint_pri_hill_painted': 5.016932823413244,
    'paint_pri_unpainted': 7.498311080253587,
    'backtrack_penalty': 28.133929949508612,
    'powerup_threshold': 79.61311396159796,
    'dominance_ratio': 0.8129998492641879,
    'bid_amount': 9.159760465061439,
    'roam_distance_penalty': 0.8600146348795171,
}
POWERUP_THRESHOLD = _W["powerup_threshold"]
DOMINANCE_RATIO   = _W["dominance_ratio"]
BID_AMOUNT        = int(_W["bid_amount"])
MEMORY            = 6
BACKTRACK_PENALTY = _W["backtrack_penalty"]

class PlayerController:
    def __init__(self, player_parity, time_left):
        self.last_loc = None
        self.history  = deque(maxlen=MEMORY)
        self.turn     = 0

    def bid(self, board, player_parity, time_left):
        return BID_AMOUNT

    def play(self, board, player_parity, time_left):
        self.turn += 1
        pb = PlayerBoard(board, player_parity)
        player = pb.get_player()
        loc, stamina = player.loc, player.stamina
        all_moves = pb.get_valid_non_beacon_moves()
        if not all_moves: return []
        opp_reachable = self._opp_reachable(pb)
        safe   = [m for m in all_moves if self._is_safe(pb, loc, m, opp_reachable)]
        safe_nb= [m for m in safe if self._dest(loc, m) != self.last_loc]
        working= safe_nb if safe_nb else (safe if safe else all_moves)
        actions= []
        m1 = self._pick(pb, working, opp_reachable, loc, stamina)
        actions.append(m1)
        pb1 = pb.get_copy(); pb1.apply_action(m1, moves_this_turn=0)
        p1  = pb1.get_player(); loc1, st1 = p1.loc, p1.stamina
        if pb1.board.cells[loc1.r][loc1.c].powerup:
            st1 = min(st1 + GameConstants.STAMINA_POWERUP_AMOUNT, p1.max_stamina)
        paints, st1 = self._paint_adj(pb1, loc1, st1); actions.extend(paints)
        self.history.append(loc); self.last_loc = loc
        cost2 = GameConstants.EXTRA_MOVE_COST
        if st1 >= cost2 + GameConstants.PAINT_STAMINA_COST:
            am2 = pb1.get_valid_non_beacon_moves(moves_this_turn=1)
            or2 = self._opp_reachable(pb1)
            s2  = [m for m in am2 if self._is_safe(pb1, loc1, m, or2)]
            s2nb= [m for m in s2 if self._dest(loc1, m) != loc]
            w2  = s2nb if s2nb else (s2 if s2 else am2)
            if w2:
                m2 = self._pick(pb1, w2, or2, loc1, st1); actions.append(m2)
                pb2= pb1.get_copy(); pb2.apply_action(m2, moves_this_turn=1)
                p2 = pb2.get_player(); loc2, st2 = p2.loc, p2.stamina - cost2
                if pb2.board.cells[loc2.r][loc2.c].powerup:
                    st2 = min(st2 + GameConstants.STAMINA_POWERUP_AMOUNT, p2.max_stamina)
                p2a, _ = self._paint_adj(pb2, loc2, st2); actions.extend(p2a)
        return actions

    def _dest(self, loc, m):
        dr, dc = m.direction.value; return Location(loc.r+dr, loc.c+dc)
    def _we_own(self, pb, loc):
        c = pb.board.cells[loc.r][loc.c]
        return c.paint_value != 0 and Parity.owned(c.paint_value, pb.player_parity)
    def _they_own(self, pb, loc):
        c = pb.board.cells[loc.r][loc.c]
        return c.paint_value != 0 and Parity.owned(c.paint_value, pb.opponent_parity)
    def _is_safe(self, pb, loc, m, opp_reachable):
        nl = self._dest(loc, m)
        return self._we_own(pb, nl) or nl not in opp_reachable
    def _opp_reachable(self, pb):
        opp = pb.get_opponent(); best = {opp.loc: opp.stamina}
        q = deque([(opp.loc, opp.stamina, 0)]); reachable = set()
        while q:
            cur, st, mu = q.popleft()
            if best.get(cur,-1) > st: continue
            nc = GameConstants.EXTRA_MOVE_COST * mu
            if st < nc: continue
            sta = st - nc
            for nb in cur.neighbors():
                if pb.board.oob(nb) or pb.board.cells[nb.r][nb.c].is_wall: continue
                if sta > best.get(nb,-1):
                    best[nb]=sta; reachable.add(nb); q.append((nb,sta,mu+1))
        return reachable
    def _pick(self, pb, moves, opp_reachable, loc, stamina):
        hills = self._hills(pb); total = len(hills)
        their_cnt = sum(1 for h in hills.values() if h["they_control"])
        for m in moves:
            nl = self._dest(loc, m)
            if nl not in opp_reachable: continue
            cell = pb.board.cells[nl.r][nl.c]
            if self._we_own(pb,nl) or cell.paint_value==0: return m
        if total > 0 and their_cnt/total >= DOMINANCE_RATIO:
            tgts = {l for h in hills.values() if h["they_control"] for l in h["cells"] if self._they_own(pb,l)}
            if tgts:
                m = self._bfs(pb, loc, moves, lambda n: n in tgts)
                if m: return m
        uph = {l for h in hills.values() for l in h["cells"] if not self._we_own(pb,l)}
        if uph:
            m = self._bfs(pb, loc, moves, lambda n: n in uph)
            if m: return m
        if stamina < POWERUP_THRESHOLD:
            m = self._bfs(pb, loc, moves, lambda n: bool(pb.board.cells[n.r][n.c].powerup))
            if m: return m
        m = self._bfs(pb, loc, moves, lambda n: pb.board.cells[n.r][n.c].paint_value==0)
        if m: return m
        return self._roam(pb, loc, moves)
    def _hills(self, pb):
        hills = {}; rows,cols = pb.board.board_size.r, pb.board.board_size.c
        thresh = GameConstants.HILL_CONTROL_THRESHOLD
        for r in range(rows):
            for c in range(cols):
                cell = pb.board.cells[r][c]; hid = cell.hill_id
                if not hid: continue
                if hid not in hills: hills[hid]={"cells":[],"ours":0,"theirs":0}
                l = Location(r,c); hills[hid]["cells"].append(l)
                if self._we_own(pb,l): hills[hid]["ours"]+=1
                elif self._they_own(pb,l): hills[hid]["theirs"]+=1
        for h in hills.values():
            t=len(h["cells"])
            h["we_control"]  =h["ours"]  >h["theirs"] and h["ours"]  /t>=thresh
            h["they_control"]=h["theirs"]>h["ours"]   and h["theirs"]/t>=thresh
        return hills
    def _bfs(self, pb, start, moves, pred):
        mm = {}
        for m in moves:
            nl=self._dest(start,m)
            if not pb.board.oob(nl) and not pb.board.cells[nl.r][nl.c].is_wall: mm.setdefault(nl,m)
        visited={start}; q=deque()
        for nl,m in mm.items():
            if pred(nl): return m
            visited.add(nl); q.append((nl,m))
        while q:
            cur,fm=q.popleft()
            for n in cur.neighbors():
                if n in visited or pb.board.oob(n) or pb.board.cells[n.r][n.c].is_wall: continue
                visited.add(n)
                if pred(n): return fm
                q.append((n,fm))
        return None
    def _roam(self, pb, loc, moves):
        mm={}
        for m in moves:
            nl=self._dest(loc,m)
            if not pb.board.oob(nl) and not pb.board.cells[nl.r][nl.c].is_wall: mm.setdefault(nl,m)
        if not mm: return None
        visited={loc}; q=deque(); cd={}
        for nl,m in mm.items(): visited.add(nl); cd[nl]=(m,1); q.append((nl,m,1))
        while q:
            cur,fm,d=q.popleft()
            for n in cur.neighbors():
                if n in visited or pb.board.oob(n) or pb.board.cells[n.r][n.c].is_wall: continue
                visited.add(n); cd[n]=(fm,d+1); q.append((n,fm,d+1))
        recent=set(self.history); best_s,best_m=-float("inf"),None
        for nl,(fm,d) in cd.items():
            s=self._cell_score(pb,nl)-d*_W["roam_distance_penalty"]
            if nl in recent: s-=BACKTRACK_PENALTY
            if s>best_s: best_s,best_m=s,fm
        return best_m
    def _cell_score(self, pb, loc):
        cell=pb.board.cells[loc.r][loc.c]; is_hill=bool(cell.hill_id)
        ours=self._we_own(pb,loc); theirs=self._they_own(pb,loc); unpainted=cell.paint_value==0
        if   is_hill and unpainted: score=_W["score_hill_unpainted"]
        elif is_hill and theirs:    score=_W["score_hill_theirs"]
        elif is_hill and not ours:  score=_W["score_hill_not_ours"]
        elif is_hill:               score=_W["score_hill_ours"]
        elif unpainted:             score=_W["score_unpainted"]
        elif theirs:                score=_W["score_theirs"]
        else:                       score=_W["score_ours"]
        if cell.powerup:                           score+=_W["score_powerup_bonus"]
        if self.last_loc and loc==self.last_loc:   score+=_W["score_last_loc_penalty"]
        return score
    def _paint_adj(self, pb, loc, stamina):
        cands=[]
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            n=Location(loc.r+dr,loc.c+dc)
            if pb.board.oob(n): continue
            cell=pb.board.cells[n.r][n.c]
            if cell.is_wall or cell.beacon_parity!=0: continue
            if self._they_own(pb,n): continue
            if self._we_own(pb,n) and abs(cell.paint_value)>=GameConstants.MAX_PAINT_VALUE: continue
            pri=(int(_W["paint_pri_hill_unpainted"]) if cell.hill_id and cell.paint_value==0 else
                 int(_W["paint_pri_hill_painted"])   if cell.hill_id else
                 int(_W["paint_pri_unpainted"])       if cell.paint_value==0 else
                 max(0,4-abs(cell.paint_value)))
            cands.append((pri,len(cands),n))
        cands.sort(reverse=True); actions=[]
        for _,_i,n in cands:
            if stamina<GameConstants.PAINT_STAMINA_COST: break
            actions.append(Action.Paint(n)); stamina-=GameConstants.PAINT_STAMINA_COST
            pb.apply_action(Action.Paint(n))
        return actions, stamina
    def commentate(self, board, player_parity, time_left):
        pb=PlayerBoard(board,player_parity); hills=self._hills(pb)
        o=sum(1 for h in hills.values() if h["we_control"])
        t=sum(1 for h in hills.values() if h["they_control"])
        return f"T{self.turn} hills={o}v{t}/{len(hills)} stamina={pb.get_player().stamina}"
