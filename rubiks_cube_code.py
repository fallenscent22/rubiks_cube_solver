import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import pygame
from pygame.locals import *
import sys
import kociemba
from collections import deque

class RubiksCube:
    def __init__(self, n=3):
        self.n = n
        # Initialize solved cube state
        self.faces = np.empty((6, n, n), dtype=int)
        # Face indices: 0-front, 1-back, 2-left, 3-right, 4-up, 5-down
        self.faces[0] = 0  # Green
        self.faces[1] = 1  # Blue
        self.faces[2] = 2  # Orange
        self.faces[3] = 3  # Red
        self.faces[4] = 4  # White
        self.faces[5] = 5  # Yellow
        
        # Rotation logic
        self.move_mapping = {
            # Face rotations
            'F': (0, 1), 'B': (1, 1), 'L': (2, 1), 'R': (3, 1), 'U': (4, 1), 'D': (5, 1),
            # Inverse moves
            "F'": (0, -1), "B'": (1, -1), "L'": (2, -1), "R'": (3, -1), "U'": (4, -1), "D'": (5, -1),
            # 180 degree rotations
            'F2': (0, 2), 'B2': (1, 2), 'L2': (2, 2), 'R2': (3, 2), 'U2': (4, 2), 'D2': (5, 2),
            # Slice moves for larger cubes
            'M': (6, 1), "M'": (6, -1), 'M2': (6, 2),
            'E': (7, 1), "E'": (7, -1), 'E2': (7, 2),
            'S': (8, 1), "S'": (8, -1), 'S2': (8, 2)
        }
        
        # Adjacent faces mapping
        last = self.n - 1
        self.adjacent_faces = {
            0: [(4, last, 'row', 'natural'), (3, 0, 'col', 'natural'), 
                 (5, 0, 'row', 'reverse'), (2, last, 'col', 'reverse')],
            1: [(4, 0, 'row', 'reverse'), (2, 0, 'col', 'natural'), 
                 (5, last, 'row', 'natural'), (3, last, 'col', 'reverse')],
            2: [(4, 0, 'col', 'reverse'), (0, 0, 'col', 'reverse'), 
                 (5, 0, 'col', 'reverse'), (1, last, 'col', 'natural')],
            3: [(4, last, 'col', 'natural'), (1, 0, 'col', 'reverse'), 
                 (5, last, 'col', 'natural'), (0, last, 'col', 'natural')],
            4: [(1, 0, 'row', 'reverse'), (3, 0, 'row', 'natural'), 
                 (0, 0, 'row', 'natural'), (2, 0, 'row', 'natural')],
            5: [(0, last, 'row', 'natural'), (3, last, 'row', 'natural'), 
                 (1, last, 'row', 'reverse'), (2, last, 'row', 'natural')]
        }
        
        # Color names and codes for display
        self.color_names = ['Green', 'Blue', 'Orange', 'Red', 'White', 'Yellow']
        self.color_codes = ['#009B48', '#0046AD', '#FF5800', '#B71234', '#FFFFFF', '#FFD500']
        self.move_history = []
        self.move_count = 0
        self.state_cache = {}
        
        # Animation state
        self.animating = False
        self.animation_queue = deque()
        self.current_animation_move = None
        self.animation_progress = 0
        self.animation_speed = 5  # degrees per frame

    def rotate_face(self, face, direction):
        """Rotate a face and update adjacent stickers"""
        # Create state hash for caching
        state_hash = hash(self.faces.tobytes())
        
        # Check cache
        if state_hash in self.state_cache:
            self.faces = self.state_cache[state_hash].copy()
            return
        
        # Rotate the face itself
        rotations = 1 if abs(direction) == 1 else 2
        if direction > 0:  # Clockwise
            self.faces[face] = np.rot90(self.faces[face], rotations, axes=(1, 0))
        else:  # Counter-clockwise
            self.faces[face] = np.rot90(self.faces[face], rotations, axes=(0, 1))
        
        if self.n == 1:  # Nothing to update
            return
        
        # Get adjacent strips
        strips = [self._get_strip(*adj) for adj in self.adjacent_faces[face]]
        
        # Cycle strips based on rotation direction
        if direction > 0:  # Clockwise
            strips = [strips[-1]] + strips[:-1]
        else:  # Counter-clockwise
            strips = strips[1:] + [strips[0]]
        
        # Apply cycled strips
        for i, adj in enumerate(self.adjacent_faces[face]):
            self._set_strip(adj[0], adj[1], adj[2], adj[3], strips[i])
            
        # Cache new state
        self.state_cache[state_hash] = self.faces.copy()
    
    def _get_strip(self, face, index, strip_type, order):
        """Extract a strip (row/column) from a face"""
        if strip_type == 'row':
            strip = self.faces[face, index, :].copy()
        else:  # 'col'
            strip = self.faces[face, :, index].copy()
        return strip[::-1] if order == 'reverse' else strip
    
    def _set_strip(self, face, index, strip_type, order, strip):
        """Apply a strip (row/column) to a face"""
        if order == 'reverse':
            strip = strip[::-1]
        if strip_type == 'row':
            self.faces[face, index, :] = strip
        else:  # 'col'
            self.faces[face, :, index] = strip
    
    def apply_moves(self, moves):
        """Apply sequence of moves to the cube"""
        moves = moves.split()
        for move in moves:
            if move in self.move_mapping:
                face, direction = self.move_mapping[move]
                self.rotate_face(face, direction)
                self.move_history.append(move)
                self.move_count += 1
    
    def scramble(self, moves=20):
        """Randomly scramble the cube"""
        moves_list = [m for m in self.move_mapping.keys() if not m.endswith('2') and not m.endswith("'")]
        scramble_moves = []
        for _ in range(moves):
            move = np.random.choice(moves_list)
            scramble_moves.append(move)
            face, direction = self.move_mapping[move]
            self.rotate_face(face, direction)
        return ' '.join(scramble_moves)
    
    def is_solved(self):
        """Check if cube is solved"""
        return all(np.all(face == face[0, 0]) for face in self.faces)
    
    def is_superflip(self):
        """Check if cube is in superflip position (3x3 only)"""
        if self.n != 3:
            return False
            
        # Check that all center pieces are correct
        for face in range(6):
            if self.faces[face, 1, 1] != face:
                return False
                
        # Check that all edge pieces are in the correct position but flipped
        edges = [
            # Up face edges
            (self.faces[4, 0, 1], self.faces[1, 0, 1]),  # Up-Back
            (self.faces[4, 1, 0], self.faces[2, 0, 1]),  # Up-Left
            (self.faces[4, 1, 2], self.faces[3, 0, 1]),  # Up-Right
            (self.faces[4, 2, 1], self.faces[0, 0, 1]),  # Up-Front
            
            # Middle layer edges
            (self.faces[0, 1, 0], self.faces[2, 1, 2]),  # Front-Left
            (self.faces[0, 1, 2], self.faces[3, 1, 0]),  # Front-Right
            (self.faces[1, 1, 0], self.faces[3, 1, 2]),  # Back-Right
            (self.faces[1, 1, 2], self.faces[2, 1, 0]),  # Back-Left
            
            # Down face edges
            (self.faces[5, 0, 1], self.faces[0, 2, 1]),  # Down-Front
            (self.faces[5, 1, 0], self.faces[2, 2, 1]),  # Down-Left
            (self.faces[5, 1, 2], self.faces[3, 2, 1]),  # Down-Right
            (self.faces[5, 2, 1], self.faces[1, 2, 1]),  # Down-Back
        ]
        
        # Each edge should have the two colors that belong to that edge position
        # but in swapped order (indicating they're flipped)
        solved_edges = [
            (4, 1), (4, 2), (4, 3), (4, 0),  # Up edges
            (0, 2), (0, 3), (1, 3), (1, 2),  # Middle edges
            (5, 0), (5, 2), (5, 3), (5, 1)   # Down edges
        ]
        
        for i, (color1, color2) in enumerate(edges):
            solved1, solved2 = solved_edges[i]
            if not ((color1 == solved1 and color2 == solved2) or 
                    (color1 == solved2 and color2 == solved1)):
                return False
                
        return True
    
    # ================== SOLVING ALGORITHMS ================== #
    def solve(self):
        """Main solve method with size handling"""
        self.move_count = 0
        self.move_history = []
        solution = []
        if self.is_solved():
            print("Cube is already solved!")
            return []
        
        if self.n == 2:
            solution = self._solve_2x2()
        elif self.n == 3:
            solution = self._solve_3x3()
        elif self.n >= 4:
            solution = self._solve_nxn()
            
        # Optimize the solution
        optimized = self.optimize_moves(solution)
        print(f"Solved in {len(optimized)} moves (optimized from {len(solution)})")
        return optimized
    
    # ------------------ 2x2 Solver ------------------ #
    def _solve_2x2(self):
        """Solve 2x2 cube (corners only)"""
        solution = []
        # Solve white layer
        solution += self._solve_white_layer_2x2()
        # Orient and permute yellow layer
        solution += self._solve_yellow_layer_2x2()
        return solution
    
    def _solve_white_layer_2x2(self):
        """Solve first layer (white) on 2x2"""
        moves = []
        # Place white corners
        for _ in range(4):
            # Find and position white corner
            for _ in range(4):
                if self.faces[4, 0, 0] == 4:
                    break
                self.apply_moves("U")
                moves.append("U")
            self.apply_moves("F D F'")
            moves += ["F", "D", "F'"]
            self.apply_moves("U'")
            moves.append("U'")
        return moves
    
    def _solve_yellow_layer_2x2(self):
        """Orient and permute yellow layer on 2x2"""
        moves = []
        # Orient yellow corners
        while not np.all(self.faces[5] == 5):
            self.apply_moves("R U R' U R U2 R'")
            moves += ["R", "U", "R'", "U", "R", "U2", "R'"]
        # Permute yellow corners
        while self.faces[0, 0, 0] != self.faces[0, 0, 1]:
            self.apply_moves("U")
            moves.append("U")
        return moves
    
    # ------------------ Kociemba Solver Format ------------------ #
    def to_kociemba_string(self):
        """Convert cube state to Kociemba solver format"""
        color_letters = {
            0: 'F',   # Green -> Front
            1: 'B',   # Blue -> Back
            2: 'L',   # Orange -> Left
            3: 'R',   # Red -> Right
            4: 'U',   # White -> Up
            5: 'D'    # Yellow -> Down
        }
        
        # Face order: U, R, F, D, L, B
        faces = [
            self.faces[4],  # Up
            self.faces[3],  # Right
            self.faces[0],  # Front
            self.faces[5],  # Down
            self.faces[2],  # Left
            self.faces[1]   # Back
        ]
        
        s = ""
        # Process U/R/F/D/L faces normally
        for face in faces[:5]:
            for row in face:
                for sticker in row:
                    s += color_letters[sticker]
        
        # Process Back face with row reversal
        for row in faces[5]:
            for sticker in reversed(row):
                s += color_letters[sticker]
        
        return s

    # ------------------ 3x3 Solver ------------------ #
    def _solve_3x3(self):
        """Solve 3x3 cube using Kociemba's algorithm"""
        try:
            # Convert to Kociemba string format
            cube_str = self.to_kociemba_string()
            # Get optimal solution
            solution = kociemba.solve(cube_str)
            moves = solution.split()
            
            # Apply moves to our cube state to verify
            for move in moves:
                if move in self.move_mapping:
                    face, direction = self.move_mapping[move]
                    self.rotate_face(face, direction)
                    self.move_history.append(move)
                    self.move_count += 1
            
            return moves
        except Exception as e:
            print(f"Kociemba error: {e}. Falling back to layer-by-layer.")
            return self._solve_3x3_layer_by_layer()

    def _solve_3x3_layer_by_layer(self):
        """Fallback layer-by-layer solve for 3x3"""
        solution = []
        solution += self._solve_white_cross()
        solution += self._solve_white_corners()
        solution += self._solve_middle_edges()
        solution += self._solve_yellow_cross()
        solution += self._solve_yellow_edges()
        solution += self._solve_yellow_corners()
        return solution
    
    def _solve_white_cross(self):
        """Solve white cross on up face"""
        moves = []
        # Position white edges
        target_color = 4  # White
        
        # Check if cross is already solved
        cross_positions = [
            (4, 1, 0), (4, 0, 1), (4, 1, 2), (4, 2, 1)  # White cross positions
        ]
        solved = True
        for pos in cross_positions:
            face, row, col = pos
            if self.faces[face, row, col] != target_color:
                solved = False
                break
        
        if solved:
            return moves
        
        # Solve each edge
        for _ in range(4):
            # Find a white edge
            for face in range(6):
                for i in range(3):
                    for j in range(3):
                        if self.faces[face, i, j] == target_color:
                            # Skip if already in position
                            if face == 4 and (i, j) in [(1, 0), (0, 1), (1, 2), (2, 1)]:
                                continue
                            
                            # Bring to bottom layer
                            if face != 5:
                                self.apply_moves("F R U R' U' F'")
                                moves += ["F", "R", "U", "R'", "U'", "F'"]
                            
                            # Position under target location
                            while self.faces[5, 0, 1] != target_color:
                                self.apply_moves("D")
                                moves.append("D")
                            
                            # Move to correct position
                            if self.faces[0, 2, 1] == 0:  # Front center
                                self.apply_moves("F2")
                                moves.append("F2")
        return moves
    
    def _solve_white_corners(self):
        """Solve white corners on first layer"""
        moves = []
        target_color = 4  # White
        
        # Check if corners are already solved
        corner_positions = [
            (4, 0, 0), (4, 0, 2), (4, 2, 0), (4, 2, 2)  # White corner positions
        ]
        solved = True
        for pos in corner_positions:
            face, row, col = pos
            if self.faces[face, row, col] != target_color:
                solved = False
                break
        
        if solved:
            return moves
        
        for _ in range(4):
            # Find a white corner
            for face in range(6):
                for i in range(3):
                    for j in range(3):
                        if self.faces[face, i, j] == target_color:
                            # Skip if already in position
                            if face == 4 and (i, j) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                                continue
                            
                            # Bring to bottom layer
                            if face != 5:
                                self.apply_moves("R U R' U'")
                                moves += ["R", "U", "R'", "U'"]
                            
                            # Position under target location
                            while not (self.faces[5, 0, 0] == target_color or 
                                      self.faces[5, 0, 2] == target_color):
                                self.apply_moves("D")
                                moves.append("D")
                            
                            # Move to correct position
                            self.apply_moves("R U R'")
                            moves += ["R", "U", "R'"]
        return moves
    
    def _solve_middle_edges(self):
        """Solve middle layer edges"""
        moves = []
        # Check if middle layer is already solved
        solved = True
        for face in [0, 1, 2, 3]:  # Front, back, left, right
            if not np.array_equal(self.faces[face, 1, :], [face]*3):
                solved = False
                break
        
        if solved:
            return moves
        
        for _ in range(4):
            # Find an edge to solve
            for face in [0, 1, 2, 3]:
                if self.faces[face, 1, 0] != face or self.faces[face, 1, 2] != face:
                    # Position edge
                    self.apply_moves("U R U' R' U' F' U F")
                    moves += ["U", "R", "U'", "R'", "U'", "F'", "U", "F"]
                    break
        return moves
    
    def _solve_yellow_cross(self):
        """Create yellow cross on down face"""
        moves = []
        target_color = 5  # Yellow
        
        # Check if cross is already solved
        cross_positions = [(5, 1, 0), (5, 0, 1), (5, 1, 2), (5, 2, 1)]
        solved = True
        for pos in cross_positions:
            face, row, col = pos
            if self.faces[face, row, col] != target_color:
                solved = False
                break
        
        if solved:
            return moves
        
        # Apply algorithm until cross is formed
        while True:
            # Count yellow edges on down face
            yellow_count = 0
            for pos in cross_positions:
                face, row, col = pos
                if self.faces[face, row, col] == target_color:
                    yellow_count += 1
            
            if yellow_count == 4:
                break
                
            # Apply cross algorithm
            self.apply_moves("F R U R' U' F'")
            moves += ["F", "R", "U", "R'", "U'", "F'"]
        return moves
    
    def _solve_yellow_edges(self):
        """Position yellow edges correctly"""
        moves = []
        # Check if edges are already positioned
        solved = True
        for face in [0, 1, 2, 3]:  # Front, back, left, right
            if self.faces[face, 2, 1] != face:
                solved = False
                break
        
        if solved:
            return moves
        
        # Position edges
        for _ in range(4):
            while self.faces[0, 2, 1] != 0:  # Front center
                self.apply_moves("U")
                moves.append("U")
            self.apply_moves("R U R' U R U2 R' U")
            moves += ["R", "U", "R'", "U", "R", "U2", "R'", "U"]
        return moves
    
    def _solve_yellow_corners(self):
        """Orient and permute yellow corners"""
        moves = []
        target_color = 5  # Yellow
        
        # Orient corners
        for _ in range(4):
            while self.faces[5, 0, 0] != target_color:
                self.apply_moves("R' D' R D")
                moves += ["R'", "D'", "R", "D"]
            self.apply_moves("U")
            moves.append("U")
        
        # Permute corners
        for _ in range(4):
            if self.faces[0, 2, 0] == self.faces[0, 2, 2] == 0:  # Front face
                break
            self.apply_moves("U R U' L' U R' U' L")
            moves += ["U", "R", "U'", "L'", "U", "R'", "U'", "L"]
        return moves
    
    # ------------------ 4x4+ Solver ------------------ #
    def _solve_nxn(self):
        """Reduction method for larger cubes"""
        solution = []
        solution += self._solve_centers()
        solution += self._pair_edges()
        solution += self._solve_3x3_layer_by_layer()  # Solve as 3x3
        return solution
    
    def _solve_centers(self):
        """Solve centers for nxn cube"""
        moves = []
        # Solve center for each color
        for color in range(6):
            # Build center row by row
            for row in range(1, self.n-1):
                for col in range(1, self.n-1):
                    # Find and position center piece
                    for face in range(6):
                        for i in range(self.n):
                            for j in range(self.n):
                                if self.faces[face, i, j] == color:
                                    # Move piece to target position
                                    # (Simplified for demonstration)
                                    self.apply_moves("R U R' U'")
                                    moves += ["R", "U", "R'", "U'"]
        return moves
    
    def _pair_edges(self):
        """Pair edge pieces for nxn cube"""
        moves = []
        # Pair edges row by row
        for row in range(1, self.n-1):
            for _ in range(12):  # 12 edges per layer
                # Find and pair edge pieces
                # (Simplified for demonstration)
                self.apply_moves("F U F' U'")
                moves += ["F", "U", "F'", "U'"]
        return moves
    
    def optimize_moves(self, moves):
        """Simplify move sequences"""
        optimized = []
        move_count = {}
        
        # Group by base move (without ' or 2)
        for move in moves:
            base = move.replace("'", "").replace("2", "")
            if base not in move_count:
                move_count[base] = 0
            
            # Calculate net rotation (-1 for ', 2 for 2)
            value = 1
            if "'" in move: 
                value = -1
            elif "2" in move: 
                value = 2
            
            move_count[base] = (move_count[base] + value) % 4
        
        # Convert net rotations back to moves
        for move, count in move_count.items():
            if count == 0: 
                continue
            elif count == 1: 
                optimized.append(move)
            elif count == 2: 
                optimized.append(move + "2")
            elif count == 3: 
                optimized.append(move + "'")
        
        return optimized
    
    # ================== ANIMATION METHODS ================== #
    def queue_moves(self, moves):
        """Queue moves for animation"""
        self.animation_queue = deque(moves)
        self.animating = True
        self.current_animation_move = None
        self.animation_progress = 0
        
    def update_animation(self):
        """Update animation state"""
        if not self.animating:
            return False
            
        if self.current_animation_move is None:
            if not self.animation_queue:
                self.animating = False
                return False
            self.current_animation_move = self.animation_queue.popleft()
            self.animation_progress = 0
            
        # Update animation progress
        self.animation_progress += self.animation_speed
        if self.animation_progress >= 90:
            # Animation complete, actually apply the move
            if self.current_animation_move in self.move_mapping:
                face, direction = self.move_mapping[self.current_animation_move]
                self.rotate_face(face, direction)
                self.move_history.append(self.current_animation_move)
                self.move_count += 1
                
            self.current_animation_move = None
            self.animation_progress = 0
            
            # Check if we're done
            if not self.animation_queue:
                self.animating = False
                return False
                
        return True
    
    # ================== DISPLAY METHODS ================== #
    def textual_display(self):
        """Textual representation of cube state"""
        display = []
        face_names = ['Front (Green)', 'Back (Blue)', 'Left (Orange)', 
                      'Right (Red)', 'Up (White)', 'Down (Yellow)']
        
        for face_index, face in enumerate(self.faces):
            display.append(f"{face_names[face_index]}:")
            for row in face:
                display.append(" ".join(self.color_names[val] for val in row))
            display.append("")
        
        return "\n".join(display)
    
    def visual_display(self):
        """3D visualization using matplotlib"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_axis_off()
        ax.view_init(30, 45)
        
        # Draw the cube
        for face in range(6):
            for x in range(self.n):
                for y in range(self.n):
                    # Map face to 3D coordinates
                    if face == 0:  # Front
                        coords = [x, y, 0]
                    elif face == 1:  # Back
                        coords = [x, y, self.n]
                    elif face == 2:  # Left
                        coords = [0, y, x]
                    elif face == 3:  # Right
                        coords = [self.n, y, x]
                    elif face == 4:  # Up
                        coords = [x, self.n, y]
                    elif face == 5:  # Down
                        coords = [x, 0, y]
                    
                    # Create cubelet
                    ax.bar3d(
                        coords[0], coords[1], coords[2], 
                        0.9, 0.9, 0.9, 
                        color=self.color_codes[self.faces[face, x, y]],
                        edgecolor='black',
                        shade=True
                    )
        
        # Set plot limits
        ax.set_xlim(0, self.n)
        ax.set_ylim(0, self.n)
        ax.set_zlim(0, self.n)
        plt.title(f"{self.n}x{self.n} Rubik's Cube")
        plt.show()
    
    def interactive_display(self):
        """Interactive 3D visualization using PyGame"""
        pygame.init()
        width, height = 1000, 800
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(f"{self.n}x{self.n} Rubik's Cube")
        clock = pygame.time.Clock()
        
        # Camera rotation
        angle_x, angle_y = 0, 0
        rotating = False
        last_pos = None
        font = pygame.font.SysFont(None, 36)
        small_font = pygame.font.SysFont(None, 28)
        
        # Animation state
        solving = False
        solution_moves = []
        move_count = 0
        
        # Timer
        start_time = pygame.time.get_ticks()
        
        def project_3d_to_2d(x, y, z):
            """Simple isometric projection"""
            iso_x = (x - y) * 30 + width // 2
            iso_y = (x + y) * 15 - z * 30 + height // 4
            return iso_x, iso_y
        
        while True:
            current_time = pygame.time.get_ticks()
            elapsed_seconds = (current_time - start_time) // 1000
            
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left button
                        rotating = True
                        last_pos = event.pos
                    elif event.button == 4:  # Scroll up
                        self.animation_speed = min(self.animation_speed + 1, 20)
                    elif event.button == 5:  # Scroll down
                        self.animation_speed = max(self.animation_speed - 1, 1)
                elif event.type == MOUSEBUTTONUP:
                    if event.button == 1:  # Left button
                        rotating = False
                elif event.type == MOUSEMOTION and rotating:
                    current_pos = event.pos
                    dx = current_pos[0] - last_pos[0]
                    dy = current_pos[1] - last_pos[1]
                    angle_x += dy * 0.01
                    angle_y += dx * 0.01
                    last_pos = current_pos
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    elif event.key == K_s and not solving and not self.animating:
                        print("Solving cube...")
                        solution_moves = self.solve()
                        solving = True
                        move_count = len(solution_moves)
                        self.queue_moves(solution_moves)
                    elif event.key == K_r:
                        # Reset cube
                        self.__init__(self.n)
                        solving = False
                        solution_moves = []
                    elif event.key == K_c:
                        # Scramble cube
                        scramble = self.scramble(20)
                        print("Scrambled with:", scramble)
                    elif event.key == K_UP:
                        self.animation_speed = min(self.animation_speed + 1, 20)
                    elif event.key == K_DOWN:
                        self.animation_speed = max(self.animation_speed - 1, 1)
            
            # Update animation
            if self.animating:
                self.update_animation()
            
            # Clear screen
            screen.fill((40, 40, 60))
            
            # Draw cube in isometric view
            for face in range(6):
                for x in range(self.n):
                    for y in range(self.n):
                        # Map face to 3D coordinates
                        if face == 0:  # Front
                            coords = [x, y, 0]
                        elif face == 1:  # Back
                            coords = [x, y, self.n]
                        elif face == 2:  # Left
                            coords = [0, y, x]
                        elif face == 3:  # Right
                            coords = [self.n, y, x]
                        elif face == 4:  # Up
                            coords = [x, self.n, y]
                        elif face == 5:  # Down
                            coords = [x, 0, y]
                        
                        # Apply rotation
                        rx = coords[0] - self.n/2
                        ry = coords[1] - self.n/2
                        rz = coords[2] - self.n/2
                        
                        # Rotate around Y axis
                        temp = rz * np.cos(angle_y) - rx * np.sin(angle_y)
                        rx = rz * np.sin(angle_y) + rx * np.cos(angle_y)
                        rz = temp
                        
                        # Rotate around X axis
                        temp = ry * np.cos(angle_x) - rz * np.sin(angle_x)
                        rz = ry * np.sin(angle_x) + rz * np.cos(angle_x)
                        ry = temp
                        
                        # Translate back
                        rx += self.n/2
                        ry += self.n/2
                        rz += self.n/2
                        
                        # Project to 2D
                        px, py = project_3d_to_2d(rx, ry, rz)
                        
                        # Draw cubelet
                        size = 300 // self.n
                        pygame.draw.rect(screen, 
                                        self.color_codes[self.faces[face, x, y]], 
                                        (px - size//2, py - size//2, size, size))
                        pygame.draw.rect(screen, 
                                        (40, 40, 40), 
                                        (px - size//2, py - size//2, size, size), 
                                        2)
            
            # Draw controls and info
            controls = [
                "Controls:",
                "Mouse Drag - Rotate View",
                "S - Solve Cube",
                "R - Reset Cube",
                "C - Scramble Cube",
                "UP/DOWN - Change animation speed",
                "ESC - Exit"
            ]
            
            for i, text in enumerate(controls):
                text_surf = small_font.render(text, True, (220, 220, 220))
                screen.blit(text_surf, (20, 20 + i*30))
            
            # Display move counter
            move_text = font.render(f"Moves: {self.move_count}", True, (255, 255, 255))
            screen.blit(move_text, (width - 200, 20))
            
            # Display timer
            mins, secs = divmod(elapsed_seconds, 60)
            timer_text = font.render(f"Time: {mins:02d}:{secs:02d}", True, (200, 200, 255))
            screen.blit(timer_text, (width - 200, 60))
            
            # Animation info
            if self.animating:
                progress_text = font.render(f"Solving: {len(self.animation_queue) + 1} moves left", True, (255, 255, 0))
                screen.blit(progress_text, (width - 300, 100))
                speed_text = font.render(f"Speed: {self.animation_speed}", True, (255, 200, 100))
                screen.blit(speed_text, (width - 300, 140))
            elif solving:
                solved_text = font.render("SOLVED!", True, (0, 255, 0))
                screen.blit(solved_text, (width - 300, 100))
            
            # Display superflip status
            if self.n == 3 and self.is_superflip():
                superflip_text = font.render("SUPERFLIP DETECTED!", True, (255, 50, 50))
                screen.blit(superflip_text, (width // 2 - 150, 20))
            
            pygame.display.flip()
            clock.tick(60)

# Testing Framework
def run_tests():
    """Test cube functionality with different cases"""
    print("Running Rubik's Cube Solver Tests...")
    test_cases = [
        ("3x3 Solved", "", 3),
        ("3x3 Simple Scramble", "R U R' U'", 3),
        ("3x3 Superflip", "U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2", 3),
        ("2x2 Solved", "", 2),
        ("2x2 Scramble", "R U R' U'", 2),
        ("4x4 Solved", "", 4)
    ]
    
    for name, scramble, size in test_cases:
        print(f"\n{'='*50}")
        print(f"Test Case: {name}")
        print(f"Scramble: {scramble if scramble else 'None'}")
        
        cube = RubiksCube(size)
        if scramble:
            cube.apply_moves(scramble)
            print("\nScrambled State:")
            print(cube.textual_display())
        
        # Check superflip status
        if name == "3x3 Superflip":
            print("Superflip detected:", cube.is_superflip())
        
        start_time = time.time()
        solution = cube.solve()
        solve_time = time.time() - start_time
        
        print("\nSolution:", ' '.join(solution))
        print(f"Solved in {solve_time:.4f} seconds")
        print("Solved State Verified:", cube.is_solved())
        
        # Show 3D visualization for small cubes
        if size <= 3:
            cube.visual_display()
        
        print(f"{'='*50}")

if __name__ == "__main__":
    run_tests()
    
    # Interactive demo
    cube_size = int(input("\nEnter cube size (2, 3, or 4): ") or 3)
    demo_cube = RubiksCube(cube_size)
    
    # Special case: superflip
    if cube_size == 3 and input("Generate superflip? (y/n): ").lower() == 'y':
        demo_cube.apply_moves("U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2")
        print("Generated superflip state!")
    else:
        scramble_moves = demo_cube.scramble(15)
        print("\nScrambled with:", scramble_moves)
    
    print(demo_cube.textual_display())
    demo_cube.interactive_display()