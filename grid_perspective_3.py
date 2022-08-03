import PIL
import pylab as pl
import numpy as np
from numpy import cos, sin

DEG = np.pi / 180

H = 1200 ## number of verticla pixels
W = 1600 ## number of horzontal pixels
FocalLength = 1000 ## number of pixels to focal point
def project(points, FocalLength,
            X, Y, Z, roll, pitch, yaw):
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

    s = (orient @ points.T).T + camera_position
    doa = FocalLength * s / s[:,2,np.newaxis]

    return doa[:,:2]

roll =  36 * DEG
pitch = 0 * DEG
yaw =  0 * DEG

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

SIDE = 1000

square = np.column_stack([[-1, -1, 1, 1],
                          [-1, 1, 1, -1],
                          [0, 0, 0, 0]]) * SIDE/2


fig, ax = pl.subplots(1, figsize=(W/200, H/200))

photo = np.array(PIL.Image.open('img/OpeningPosition.jpeg'))
ax.imshow(photo, extent=(-W/2, W/2, -H/2, H/2), origin='upper')

X, Y, Z = 200, 1000, 8500
X, Y, Z = 0, 0, 10000
camera_position = [X, Y, Z]
objects = []
raw_squares = []
for i in np.arange(-3.5, 5, 1):
    for j in np.arange(-3.5, 5, 1):
        if (i + j) % 2 == 0:
            r = np.array([i * SIDE, j * SIDE, 0])
            print(i, j, r)
            raw_squares.append(r)
            
            s = (square + r)
            p = project(s, FocalLength,
                        X, Y, Z, roll, pitch, yaw)
            objects.append(ax.fill(p[:,0], p[:,1], alpha=.8, color='r')[0])

def getSquarePos3D(alg):
    x
def on_update(*args, **kw):
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
    for r, o in zip(raw_squares, objects):
        s = square + r
        p = project(s, FocalLength, X, Y, Z, roll, pitch, yaw)
        o.set_xy(p)
    
    
        
pl.axis('equal')

def onclick(event):
    global roll, pitch, yaw, Z

    if False:
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
    if event.button == 1:
        dir = 1
    else:
        dir = -1
    x, y = event.xdata, event.ydata
    if x > 0 and y > 0:
        print('roll')
        roll += dir * DEG
    if x > 0 and y < 0:
        print('pitch')
        pitch += dir * DEG
    if x < 0 and y > 0:
        print('yaw')
        yaw += dir * DEG
    if x < 0 and y < 0:
        Z += 100 * dir
        camera_position[2] = Z
    print(np.round(roll / DEG), np.round(pitch / DEG), np.round(yaw / DEG), Z)
    on_update()
    fig.canvas.draw_idle()
cid = fig.canvas.mpl_connect('button_press_event', onclick)

pl.show()
### fit params
# FocalLength, X, Y, Z, roll, pitch, yaw
