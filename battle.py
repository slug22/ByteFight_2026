"""
battle.py — Run two bots against each other across all maps in parallel
Usage: python battle.py --a simons-bot --b orsnby-bot --games 50
"""
import argparse, json, os, random, sys, subprocess
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

MAPS = ["big_spiral", "disjoint", "matrix", "maze", "spiral",
        "test_map", "the_complex", "the_temple"]


def run_one_args(args):
    game_dir, a, b, map_name, idx = args
    try:
        root = os.path.dirname(os.path.abspath(__file__))
        cmd = [sys.executable, "run_game.py",
               "--a_name", a, "--b_name", b,
               "--map_name", map_name,
               "--game_directory", os.path.basename(game_dir),
               "--no_display", "--no_clear_screen"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=root)
        combined = r.stdout + r.stderr
        if "Player A won" in combined: return 1, map_name
        if "Player B won" in combined: return -1, map_name
        return 0, map_name
    except Exception as e:
        print(f"[error] {e}")
        return 0, map_name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a",       "-a", required=True,          help="Bot A folder name")
    p.add_argument("--b",       "-b", required=True,          help="Bot B folder name")
    p.add_argument("--games",   "-g", type=int, default=20,   help="Total games (default 20)")
    p.add_argument("--workers", "-w", type=int, default=24,   help="Parallel workers (default 24)")
    p.add_argument("--bot_dir", "-d", default="newBots",      help="Folder containing bots (default newBots)")
    p.add_argument("--maps",    "-m", default=None,           help="Comma-separated map names (default all)")
    args = p.parse_args()

    root     = os.path.dirname(os.path.abspath(__file__))
    game_dir = os.path.join(root, args.bot_dir)
    map_pool = args.maps.split(",") if args.maps else MAPS

    print(f"\n⚔  {args.a} vs {args.b}  —  {args.games} games  ({args.workers} parallel)\n")

    # Build task list, alternating sides for fairness
    task_args = []
    for i in range(args.games):
        map_name = random.choice(map_pool)
        if i % 2 == 0:
            task_args.append((game_dir, args.a, args.b, map_name, i))
        else:
            task_args.append((game_dir, args.b, args.a, map_name, i))

    a_wins = 0; b_wins = 0; ties = 0
    map_stats = defaultdict(lambda: {"a": 0, "b": 0, "tie": 0})
    results = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (result, map_name) in enumerate(ex.map(run_one_args, task_args)):
            # Flip result back to a/b perspective if sides were swapped
            if i % 2 == 1:
                result = -result

            results.append((result, map_name))

            if result == 1:
                a_wins += 1
                map_stats[map_name]["a"] += 1
                tag = f"✓ {args.a}"
            elif result == -1:
                b_wins += 1
                map_stats[map_name]["b"] += 1
                tag = f"✓ {args.b}"
            else:
                ties += 1
                map_stats[map_name]["tie"] += 1
                tag = "tie"

            total = i + 1
            wr_a = 100 * a_wins / total
            print(f"  Game {total:3d}/{args.games}  [{map_name:<12s}]  {tag:<25s}"
                  f"  {args.a}: {a_wins}  {args.b}: {b_wins}  ties: {ties}  ({wr_a:.0f}%)")

    total = args.games
    print(f"\n{'='*60}")
    print(f"  {args.a:<20s}  {a_wins}/{total}  ({100*a_wins/total:.1f}%)")
    print(f"  {args.b:<20s}  {b_wins}/{total}  ({100*b_wins/total:.1f}%)")
    print(f"  Ties:                {ties}/{total}  ({100*ties/total:.1f}%)")

    print(f"\nPer-map breakdown:")
    for map_name, s in sorted(map_stats.items()):
        played = s['a'] + s['b'] + s['tie']
        if played == 0:
            continue
        print(f"  {map_name:<14s}  {args.a}: {s['a']}  {args.b}: {s['b']}  ties: {s['tie']}")

    # Overall winner
    print(f"\n{'='*60}")
    if a_wins > b_wins:
        print(f"  🏆  {args.a} wins!")
    elif b_wins > a_wins:
        print(f"  🏆  {args.b} wins!")
    else:
        print(f"  🤝  It's a draw!")


if __name__ == "__main__":
    main()