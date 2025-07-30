class RubiksCube:
    def __init__(self,n):
        self.n = n
        #initialize the cube with n layers
        self.faces = [[[i]*n for _ in range(n)] for i in range(6)]
    def rotate_face(self, face_index, direction):
        #rotate the face in specified direction such as clockwise, anticlockwise, or 180 degrees
        #update the adjacent faces accordingly
        pass
    def solve(self):
        #solving algorithm
        if self.n == 2:
            self._solve_2x2()
        elif self.n == 3:
            self._solve_3x3()
        elif self.n >= 4:
            self._solve_nxn() #reduction method (when the cube is larger than 3x3)
    def _solve_2x2(self):
        #solve first layer corners
        #update adjacent faces/ orient last layer corners
        pass
    def _solve_3x3(self):
        #layer-by-layer approach
        self._solve_white_cross()
        self._solve_white_corners()
        self._solve_middle_edges()
        self._solve_yellow_cross()
        self._solve_yellow_edges()
        self._solve_yellow_corners()
    def _solve_nxn(self):
        # when n is >=4, solve the centers first
        #pair edges
        #solve as 3x3 cube
        pass
    