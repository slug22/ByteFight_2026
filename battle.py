"""
battle.py — Run two bots against each other across all maps
Usage: python battle.py --a simons-bot --b orsnby-bot --games 50
"""
import argparse, json, os, random, io
from contextlib import redirect_stdout, redirect_stderr
from collections import defaultdict

MAPS = ["big_spiral", "disjoint", "matrix", "maze", "spiral",
        "test_map", "the_complex", "the_temple"]

def run_one(game_dir, a, b, map_name):
    try:
        from game_runner.gameplay import play_game

        _maps_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'maps.json')
        with open(_maps_path) as f:
            map_string = json.load(f)[map_name]

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            outcome = play_game(game_dir, game_dir, a, b,
                               display_game=False, clear_screen=False,
                               record=False, limit_resources=False,
                               map_string=map_string, output_stream=None)
        return outcome.result
    except Exception as e:
        print(f"[error] {e}")
        return 0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a",       "-a", required=True)
    p.add_argument("--b",       "-b", required=True)
    p.add_argument("--games",   "-g", type=int, default=20)
    p.add_argument("--bot_dir", "-d", default="newBots")
    args = p.parse_args()

    root     = os.path.dirname(os.path.abspath(__file__))
    game_dir = os.path.join(root, args.bot_dir)

    print(f"\n⚔  {args.a} vs {args.b}  —  {args.games} games\n")

    a_wins = 0; b_wins = 0; ties = 0
    map_stats = defaultdict(lambda: {"a": 0, "b": 0, "tie": 0})

    for i in range(args.games):
        map_name = random.choice(MAPS)
        # Alternate sides each game
        if i % 2 == 0:
            result = run_one(game_dir, args.a, args.b, map_name)
        else:
            result = run_one(game_dir, args.b, args.a, map_name)
            result = -result  # flip back to a/b perspective

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

    print(f"\n{'='*60}")
    print(f"  {args.a:<20s}  {a_wins}/{args.games}  ({100*a_wins/args.games:.1f}%)")
    print(f"  {args.b:<20s}  {b_wins}/{args.games}  ({100*b_wins/args.games:.1f}%)")
    print(f"  Ties:                {ties}/{args.games}")
    print(f"\nPer-map breakdown:")
    for map_name, s in sorted(map_stats.items()):
        print(f"  {map_name:<14s}  {args.a}: {s['a']}  {args.b}: {s['b']}  ties: {s['tie']}")

if __name__ == "__main__":
    main()