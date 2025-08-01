from rubikscube.cube import RubiksCube

# Create a 3x3 cube
cube = RubiksCube(3)

# Scramble the cube
scramble_moves = cube.scramble(20)
print("Scramble:", scramble_moves)

# Print scrambled state
print("Scrambled State:")
for face_idx, face in enumerate(cube.faces):
    print(f"Face {face_idx}:")
    print(face)

# Solve the cube
solution = cube.solve()
print("Solution Moves:", solution)

# Print solved state
print("Solved State:")
for face_idx, face in enumerate(cube.faces):
    print(f"Face {face_idx}:")
    print(face)

# Check if solved
print("Is Solved?", cube.is_solved())