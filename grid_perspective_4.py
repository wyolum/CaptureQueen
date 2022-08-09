from scipy.optimize import fmin, fmin_powell, minimize
import PIL
import pylab as pl
import numpy as np
from numpy import cos, sin

DEG = np.pi / 180

def get_row_col(alg):
    row = ord(alg[0]) - ord('a')
    col = int(alg[1]) - 1
    return row, col

def get_color(alg, dark='r', light='w'):
    row, col = get_row_col(alg)
    is_dark = (row + col + 1) % 2
    if alg == 'a1':
        color = 'k'
    elif is_dark:
        color = dark
    else:
        color = light
    return color
    

def get_center(alg, side):
    offset_x = 3.5 * side ## x offset to center of the board
    offset_y = 3.5 * side ## y offset to center of the board
    row, col = get_row_col(alg)
    return np.array([row * side - offset_x, col * side - offset_y, 0])

def getOrient(roll, pitch, yaw):
    Roll = np.array([[1, 0, 0],
                     [0, cos(roll), -sin(roll)],
                     [0, sin(roll),  cos(roll)]])
    Pitch = np.array([[cos(pitch), 0, -sin(pitch)],
                      [0, 1, 0],
                      [sin(pitch), 0,  cos(pitch)]])
    Yaw = np.array([[cos(yaw), -sin(yaw), 0],
                    [sin(yaw),  cos(yaw), 0],
                    [0, 0, 1]])

    orient = Yaw @ Pitch @ Roll
    return orient

def project(points, FocalLength,
            X, Y, Z, roll, pitch, yaw):
    orient = getOrient(roll, pitch, yaw)
    pos = np.array([X, Y, Z])
    doa = points - pos
    s = (orient @ doa.T).T

    # scale to make focal lenght in third column
    doa = FocalLength * s / s[:,2,np.newaxis]
    return doa[:,:2]

def invert(points2d, FocalLength,
           X, Y, Z, roll, pitch, yaw):
    orient = getOrient(roll, pitch, yaw)
    pos = np.array([X, Y, Z])
    n = len(points2d)
    doa = np.hstack([points2d, np.ones(n)[:,np.newaxis] * FocalLength])
    s = (orient.T @ doa.T).T

    doa = (orient.T @ doa.T).T + pos
    ## [x, y, 0] = pos + doa3 * l
    ## pos[2] + doa3[:,2] * l = 0
    ## l = -pos[2] / doa3[:,2]
    l = pos[2] / doa[:,2]
    points = doa * l[:,np.newaxis]
    return points

class Square:
    def __init__(self, side, alg, dark, light):
        self.alg = alg
        self.pos = get_center(alg, side)
        self.square = np.column_stack([[-1, -1, 1, 1],
                                       [-1, 1, 1, -1],
                                       [0, 0, 0, 0]]) * side / 2 + self.pos
        self.color = get_color(alg, dark, light)
        self.side = side

class ChessBoard:
    def __init__(self, side, dark, light):
        self.squares = []
        for row in 'abcdefgh':
            for col in range(1, 9):
                alg = f'{row}{col}'
                self.squares.append(Square(side, alg, dark, light))

class Camera:
    def __init__(self, focal_length, px, py, pz, ar, ap, ay):
        self.focal_length = focal_length
        self.pos = np.array([px, py, pz])
        self.rpy = np.array([ar, ap, ay])
        self.orient = getOrient(ar, ap, ay)
        #self.fig, self.ax = pl.subplots(1)
        #cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.objects = []
        
    def project(self, points):
        return project(points, self.focal_length, *self.pos, *self.rpy)
    
    def fill(self, points, ax, *args, **kw):
        p2d = self.project(points)
        self.objects.append((points, ax.fill(p2d[:,0], p2d[:,1], *args, **kw)[0]))

    def plot(self, points, ax, *args, **kw):
        p2d = self.project(points)
        self.objects.append((points, ax.plot(p2d[:,0], p2d[:,1], *args, **kw)[0]))
        
    def snapshot(self, board, ax):
        for square in board.squares:
            self.fill(square.square, ax, color=square.color, alpha=.3)
            #self.plot(square.pos[np.newaxis], ax, 'k+')

    def get_pixel_locs(self, board):
        out = {}
        for square in board.squares:
            p2d = self.project(square.pos[np.newaxis])[0]
            out[square.alg] = p2d
        return out
    
    def on_click(self, event):
        print(event)

def get_error(board, cam1, cam2):
    slack = 1
    p1 = cam1.get_pixel_locs(board)
    p2 = cam2.get_pixel_locs(board)
    out = 0
    for key in p1:
        delta = np.linalg.norm(p1[key] - p2[key])
        if delta > slack:
            out += delta
    return out
    
side = 1000
board = ChessBoard(side, 'r', 'w')
other_board = ChessBoard(side, 'g', 'w')

focal_length = 1000
px, py, pz = 0, 0, 8200
ar = 0 * DEG
ap = 0 * DEG
ay = 180 * DEG

xx = [[100, 2, 0], [4, 5, 0]]
p = project(np.array(xx), focal_length, px, py, pz, ar, ap, ay)
i = invert(p, focal_length, px, py, pz, ar, ap, ay)

print(xx)
print(i)

meas_camera = Camera(focal_length, px, py, pz, ar, ap, ay)
test_camera = Camera(focal_length, 0, -500, 5000, 20 * DEG, 5 * DEG, 5 * DEG)

def minme(xyz):
    test_camera = Camera(focal_length, xyz[0], xyz[1], xyz[2], ar, ap, ay)
    return get_error(board, meas_camera, test_camera)

def angle_pen(a):
    if -np.pi < a and a < np.pi:
        out = 0
    else:
        out = np.abs(a) - np.pi
    return out

def minmer(xyzr):
    test_camera = Camera(focal_length, xyzr[0], xyzr[1], xyzr[2], xyzr[3], ap, ay)
    out = get_error(board, meas_camera, test_camera)
    out += angle_pen(xyzr[3])
    return out

def minmerp(xyzrp):
    test_camera = Camera(focal_length, xyzrp[0], xyzrp[1], xyzrp[2], xyzrp[3], xyzrp[4], ay)
    out = get_error(board, meas_camera, test_camera)
    out += angle_pen(xyzrp[3])
    out += angle_pen(xyzrp[4])
    return out

def minmerpy(xyzrpy):
    test_camera = Camera(focal_length, xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5])
    out = get_error(board, meas_camera, test_camera)
    out += angle_pen(xyzrpy[3])
    out += angle_pen(xyzrpy[4])
    out += angle_pen(xyzrpy[5])
    return out

if False:
    ans = fmin_powell(minme, [0, 0, 8000])
    print(ans)
    ans = fmin_powell(minmer, [0, 0, 8000, 20 * DEG])
    print(ans)
    ans = fmin_powell(minmerp, [0, 0, 8000, ar, ap])
    print(ans)
guess = [-100, -300, 5000, 0 * DEG, 0 * DEG, 0 * DEG]
guss_camera = Camera(focal_length, *guess)
ans = fmin_powell(minmerpy, guess)
print(ans)
soln_camera = Camera(focal_length, *ans)

fig, ax = pl.subplots(1)
meas_camera.snapshot(board, ax)
soln_camera.snapshot(board, ax)
guss_camera.snapshot(other_board, ax)
#pl.axes('equal')
pl.show()

here
