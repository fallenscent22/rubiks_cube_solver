import argparse
from rubikscube.cube import RubiksCube
from rubikscube.visualization import CubeVisualizer
import time


def main():
    parser = argparse.ArgumentParser(description='Rubik\'s Cube Solver CLI')
    parser.add_argument('--size', type=int, default=3, help='Cube size (2, 3, or 4)')
    parser.add_argument('--scramble', type=str, default='', help='Scramble moves (e.g. "R U R\' U\'")')
    parser.add_argument('--superflip', action='store_true', help='Generate superflip state (3x3 only)')
    args = parser.parse_args()

    cube = RubiksCube(args.size)
    visualizer = CubeVisualizer(cube)

    if args.superflip and args.size == 3:
        cube.apply_moves("U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2")
        print("Generated superflip state!")
    elif args.scramble:
        cube.apply_moves(args.scramble)
        print(f"Scrambled with: {args.scramble}")
    else:
        scramble_moves = cube.scramble(20)
        print(f"Scrambled with: {scramble_moves}")

    print("\nScrambled State:")
    print(visualizer.textual_display())

    start_time = time.time()
    solution = cube.solve()
    solve_time = time.time() - start_time

    print("\nSolution:", ' '.join(solution))
    print(f"Solved in {solve_time:.4f} seconds")
    print("Solved State Verified:", cube.is_solved())
    print("\nSolved State:")
    print(visualizer.textual_display())

    if args.size <= 3:
        visualizer.visual_display()

    solution = cube.solve()
    print("Solution Moves:", solution)
    print("Solved State:")
    for face_idx, face in enumerate(cube.faces):
        print(f"Face {face_idx}:")
        print(face)
    print("Is Solved?", cube.is_solved())

if __name__ == "__main__":
    main()
