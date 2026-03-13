"""
tuner.py — Evolutionary self-play weight tuner for controller.py
Usage:
    python tuner.py --bot_name my_bot --map_name test_map
    python tuner.py --generations 50 --games_per_eval 6 --mutation_scale 0.15
"""

import argparse, copy, json, os, random, shutil, subprocess, sys, time

DEFAULT_WEIGHTS = {
    "score_hill_unpainted":     100.0,
    "score_hill_theirs":         80.0,
    "score_hill_not_ours":       60.0,
    "score_hill_ours":           20.0,
    "score_unpainted":           30.0,
    "score_theirs":              10.0,
    "score_ours":                 5.0,
    "score_powerup_bonus":       50.0,
    "score_last_loc_penalty":   -20.0,
    "paint_pri_hill_unpainted":  10.0,
    "paint_pri_hill_painted":     8.0,
    "paint_pri_unpainted":        6.0,
    "backtrack_penalty":         40.0,
    "powerup_threshold":         70.0,
    "dominance_ratio":            0.60,
    "bid_amount":                40.0,
    "roam_distance_penalty":      0.5,
}

MUTATION_SCALES = {k: 0.20 for k in DEFAULT_WEIGHTS}
MUTATION_SCALES.update({"backtrack_penalty": 0.25, "powerup_threshold": 0.15,
                         "dominance_ratio": 0.10, "score_unpainted": 0.25,
                         "score_theirs": 0.25, "score_ours": 0.25})

BOUNDS = {
    "dominance_ratio":       (0.40, 0.90),
    "powerup_threshold":     (20.0, 95.0),
    "bid_amount":            (0.0,  100.0),
    "roam_distance_penalty": (0.0,  5.0),
}


def mutate(weights, scale=1.0):
    new_w = copy.deepcopy(weights)
    chosen = random.sample(list(new_w), max(1, len(new_w) // 3))
    for k in chosen:
        new_w[k] += new_w[k] * MUTATION_SCALES.get(k, 0.20) * scale * random.gauss(0, 1)
        if k in BOUNDS:
            lo, hi = BOUNDS[k]
            new_w[k] = max(lo, min(hi, new_w[k]))
    return new_w


TEMPLATE = '''\
from collections.abc import Callable
from collections import deque
from game import *
from .player_board import PlayerBoard

_W = {weights_repr}
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
        opp = pb.get_opponent(); best = {{opp.loc: opp.stamina}}
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
            tgts = {{l for h in hills.values() if h["they_control"] for l in h["cells"] if self._they_own(pb,l)}}
            if tgts:
                m = self._bfs(pb, loc, moves, lambda n: n in tgts)
                if m: return m
        uph = {{l for h in hills.values() for l in h["cells"] if not self._we_own(pb,l)}}
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
        hills = {{}}; rows,cols = pb.board.board_size.r, pb.board.board_size.c
        thresh = GameConstants.HILL_CONTROL_THRESHOLD
        for r in range(rows):
            for c in range(cols):
                cell = pb.board.cells[r][c]; hid = cell.hill_id
                if not hid: continue
                if hid not in hills: hills[hid]={{"cells":[],"ours":0,"theirs":0}}
                l = Location(r,c); hills[hid]["cells"].append(l)
                if self._we_own(pb,l): hills[hid]["ours"]+=1
                elif self._they_own(pb,l): hills[hid]["theirs"]+=1
        for h in hills.values():
            t=len(h["cells"])
            h["we_control"]  =h["ours"]  >h["theirs"] and h["ours"]  /t>=thresh
            h["they_control"]=h["theirs"]>h["ours"]   and h["theirs"]/t>=thresh
        return hills
    def _bfs(self, pb, start, moves, pred):
        mm = {{}}
        for m in moves:
            nl=self._dest(start,m)
            if not pb.board.oob(nl) and not pb.board.cells[nl.r][nl.c].is_wall: mm.setdefault(nl,m)
        visited={{start}}; q=deque()
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
        mm={{}}
        for m in moves:
            nl=self._dest(loc,m)
            if not pb.board.oob(nl) and not pb.board.cells[nl.r][nl.c].is_wall: mm.setdefault(nl,m)
        if not mm: return None
        visited={{loc}}; q=deque(); cd={{}}
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
        return f"T{{self.turn}} hills={{o}}v{{t}}/{{len(hills)}} stamina={{pb.get_player().stamina}}"
'''


def render_controller(weights):
    w = "{\n" + "".join(f'    {k!r}: {v!r},\n' for k,v in weights.items()) + "}"
    return TEMPLATE.format(weights_repr=w)

def write_bot(bot_dir, weights, source_dir):
    os.makedirs(bot_dir, exist_ok=True)
    for fname in ["player_board.py", "__init__.py"]:
        src = os.path.join(source_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(bot_dir, fname))
    with open(os.path.join(bot_dir, "controller.py"), "w") as f:
        f.write(render_controller(weights))

def run_one_game(run_game_py, game_dir, a, b, map_name, out):
    cmd = [sys.executable, run_game_py,
           "--a_name", a, "--b_name", b,
           "--map_name", map_name, "--game_directory", game_dir,
           "--output_file", out, "--no_display", "--no_clear_screen"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        combined = r.stdout + r.stderr   # <-- check both streams
        if "Player A won" in combined: return 1
        if "Player B won" in combined: return -1
        return 0
    except Exception as e:
        print(f"[game error] {e}")
        return 0

from concurrent.futures import ProcessPoolExecutor
def evaluate(run_game_py, game_dir, champ, chall, map_name, n_games, match_dir):
    args = []
    for i in range(n_games):
        out = os.path.join(match_dir, f"g{i}.json")
        if i % 2 == 0:
            args.append((run_game_py, game_dir, chall, champ, map_name, out, i))
        else:
            args.append((run_game_py, game_dir, champ, chall, map_name, out, i))

    with ProcessPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(_run_game_worker, args))

    score = 0.0
    for i, r in enumerate(results):
        if i % 2 == 0:
            score += 1.0 if r == 1 else (0.5 if r == 0 else 0)
        else:
            score += 1.0 if r == -1 else (0.5 if r == 0 else 0)
    return score / n_games

def _run_game_worker(args):
    run_game_py, game_dir, a, b, map_name, out, idx = args
    return run_one_game(run_game_py, game_dir, a, b, map_name, out)
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bot_name",       "-n", default="sample_controller")
    p.add_argument("--game_directory", "-g", default="workspace")
    p.add_argument("--map_name",       "-m", default="test_map")
    p.add_argument("--generations",    "-G", type=int,   default=30)
    p.add_argument("--games_per_eval", "-e", type=int,   default=4)
    p.add_argument("--mutation_scale", "-s", type=float, default=1.0)
    p.add_argument("--win_threshold",  "-t", type=float, default=0.55)
    p.add_argument("--weights_file",   "-w", default="tuner_weights.json")
    p.add_argument("--run_game_py",    "-r", default="run_game.py")
    args = p.parse_args()

    root       = os.path.dirname(os.path.abspath(args.run_game_py))
    game_dir   = os.path.join(root, args.game_directory)
    source_bot = os.path.join(game_dir, args.bot_name)
    wfile      = os.path.join(root, args.weights_file)
    log_file   = os.path.join(root, "tuner_log.json")
    match_dir  = os.path.join(root, "tuner_matches")
    os.makedirs(match_dir, exist_ok=True)

    champ_name = args.bot_name + "_champ"
    chall_name = args.bot_name + "_chall"
    champ_dir  = os.path.join(game_dir, champ_name)
    chall_dir  = os.path.join(game_dir, chall_name)

    if os.path.exists(wfile):
        with open(wfile) as f: champ_w = json.load(f)
        for k, v in DEFAULT_WEIGHTS.items(): champ_w.setdefault(k, v)
        print(f"[tuner] Loaded weights from {wfile}")
    else:
        champ_w = copy.deepcopy(DEFAULT_WEIGHTS)
        print("[tuner] Starting from defaults")

    log = []; promotions = 0
    print(f"\n[tuner] {args.generations} gens | {args.games_per_eval} games/eval | "
          f"threshold {args.win_threshold:.0%} | scale {args.mutation_scale}\n")

    for gen in range(1, args.generations + 1):
        t0      = time.time()
        chall_w = mutate(champ_w, args.mutation_scale)
        write_bot(champ_dir, champ_w, source_bot)
        write_bot(chall_dir, chall_w, source_bot)
        wr       = evaluate(args.run_game_py, game_dir, champ_name, chall_name,
                            args.map_name, args.games_per_eval, match_dir)
        elapsed  = time.time() - t0
        promoted = wr >= args.win_threshold
        tag = "✓ PROMOTED" if promoted else "  rejected"
        print(f"  Gen {gen:3d}/{args.generations}  wr={wr:.2f}  {tag}  ({elapsed:.1f}s)")
        if promoted:
            promotions += 1; champ_w = chall_w
            with open(wfile, "w") as f: json.dump(champ_w, f, indent=2)
            print(f"            → saved to {wfile}")
        log.append({"gen": gen, "wr": wr, "promoted": promoted, "elapsed": round(elapsed,2)})
        with open(log_file, "w") as f: json.dump(log, f, indent=2)

    with open(wfile, "w") as f: json.dump(champ_w, f, indent=2)
    print(f"\n[tuner] Done. {promotions}/{args.generations} promotions.")
    print(f"        Weights → {wfile}  |  Log → {log_file}")
    print("\nFinal weights vs defaults:")
    for k, v in champ_w.items():
        d = v - DEFAULT_WEIGHTS.get(k, v)
        arrow = f"  ({'+' if d>=0 else ''}{d:.2f})" if abs(d) > 0.01 else ""
        print(f"  {k:<30s} {v:8.2f}{arrow}")

    try:
        ans = input("\nApply back to source bot? [y/N] ").strip().lower()
    except EOFError:
        ans = "n"
    if ans == "y":
        bak = source_bot + "_backup"
        shutil.copytree(source_bot, bak, dirs_exist_ok=True)
        with open(os.path.join(source_bot, "controller.py"), "w") as f:
            f.write(render_controller(champ_w))
        print(f"  Saved! (original backed up to {bak}/)")

    for d in [champ_dir, chall_dir]:
        if os.path.exists(d): shutil.rmtree(d, ignore_errors=True)

if __name__ == "__main__":
    main()