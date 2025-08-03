import random
import time
import sys
from Rubiks_Cube_Solver import cube_solver
from Rubiks_Cube_Solver.cube_model import Cube
from Rubiks_Cube_Solver.cube_solver import Solver
from Rubiks_Cube_Solver.move_optimizer import optimize_moves
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

SOLVED_CUBE_STR = "OOOOOOOOOYYYWWWGGGBBBYYYWWWGGGBBBYYYWWWGGGBBBRRRRRRRRR"
MOVES = ["L", "R", "U", "D", "F", "B", "M", "E", "S"]


def random_cube_model():
    """
    :return: A new scrambled Cube
    """
    scramble_moves = " ".join(random.choices(MOVES, k=200))
    a = Cube(SOLVED_CUBE_STR)
    a.sequence(scramble_moves)
    return a


def run(max_solves=None, save_file="solver_stats.txt"):
    successes = 0
    failures = 0
    avg_opt_moves = 0.0
    avg_moves = 0.0
    avg_time = 0.0
    total = 0
    bar = None
    if max_solves and tqdm:
        bar = tqdm(total=max_solves, desc="Solving Cubes")
    try:
        while True:
            if max_solves and total >= max_solves:
                break
            C = random_cube_model()
            cube_solverr = Solver(C)
            start = time.time()
            cube_solverr.solve()
            duration = time.time() - start
            if C.is_solved():
                opt_moves = optimize_moves(cube_solverr.moves)
                successes += 1
                avg_moves = (avg_moves * (successes - 1) + len(cube_solverr.moves)) / float(successes)
                avg_time = (avg_time * (successes - 1) + duration) / float(successes)
                avg_opt_moves = (avg_opt_moves * (successes - 1) + len(opt_moves)) / float(successes)
            else:
                failures += 1
                print(f"Failed ({successes + failures}): {C.flat_str()}")
            total = successes + failures
            if bar:
                bar.update(1)
            if total == 1 or total % 100 == 0:
                pass_percentage = 100 * successes / total
                print(f"{total}: {successes} successes ({pass_percentage:0.3f}% passing)"
                      f" avg_moves={avg_moves:0.3f} avg_opt_moves={avg_opt_moves:0.3f}"
                      f" avg_time={avg_time:0.3f}s")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving stats...")
        stats = {
            "successes": successes,
            "failures": failures,
            "avg_moves": avg_moves,
            "avg_opt_moves": avg_opt_moves,
            "avg_time": avg_time,
            "total": total
        }
        with open(save_file, "w") as f:
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")
        print(f"Stats saved to {save_file}")
    if bar:
        bar.close()


if __name__ == '__main__':
    cube_solver.DEBUG = False
    import argparse
    parser = argparse.ArgumentParser(description="Rubik's Cube random solver runner")
    parser.add_argument('--max_solves', type=int, default=None, help='Maximum number of solves to run')
    parser.add_argument('--save_file', type=str, default='solver_stats.txt', help='File to save stats on exit')
    args = parser.parse_args()
    run(max_solves=args.max_solves, save_file=args.save_file)
