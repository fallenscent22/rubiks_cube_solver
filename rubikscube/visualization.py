import matplotlib.pyplot as plt
import numpy as np

class CubeVisualizer:
    def __init__(self, cube):
        self.cube = cube

    def textual_display(self):
        display = []
        face_names = ['Front (Green)', 'Back (Blue)', 'Left (Orange)', 
                      'Right (Red)', 'Up (White)', 'Down (Yellow)']
        for face_index, face in enumerate(self.cube.faces):
            display.append(f"{face_names[face_index]}:")
            for row in face:
                display.append(" ".join(self.cube.color_names[val] for val in row))
            display.append("")
        return "\n".join(display)

    def visual_display(self):
        n = self.cube.n
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_axis_off()
        ax.view_init(30, 45)
        for face in range(6):
            for x in range(n):
                for y in range(n):
                    if face == 0:
                        coords = [x, y, 0]
                    elif face == 1:
                        coords = [x, y, n]
                    elif face == 2:
                        coords = [0, y, x]
                    elif face == 3:
                        coords = [n, y, x]
                    elif face == 4:
                        coords = [x, n, y]
                    elif face == 5:
                        coords = [x, 0, y]
                    ax.bar3d(
                        coords[0], coords[1], coords[2], 
                        0.9, 0.9, 0.9, 
                        color=self.cube.color_codes[self.cube.faces[face, x, y]],
                        edgecolor='black',
                        shade=True
                    )
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_zlim(0, n)
        plt.title(f"{n}x{n} Rubik's Cube")
        plt.show()
