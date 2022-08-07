from scipy.optimize import fmin, fmin_powell, minimize
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

positions = {'a1':np.array((-636, -374)),
             'a8':np.array([-376.38139061,320.14850568]),
             'h8':np.array([311.20660502, 317.08544763]),
             'h1':np.array([532, -363]),
             'a6': np.array([-426.45396426,188.16996774]),
             'h6':np.array([354.62282089, 185.72452119]),
             'h7':np.array([331.53872035, 255.56821376]),
             'd1':np.array([-132, -374]),
             'b7':np.array([-294.6754694,257.97434045]),
             'e5':np.array([23, 106])
}


X, Y, Z = -300, 500, 8200
X, Y, Z = 0, 500, 8200
camera_position = [X, Y, Z]
objects = []
raw_squares = []
offset_x = 3.5 * SIDE
offset_y = 3.5 * SIDE


def get_row_col(alg):
    row = ord(alg[0]) - ord('a')
    col = int(alg[1]) - 1
    return row, col

def get_color(alg):
    row, col = get_row_col(alg)
    return (row + col) % 2

def get_center(alg):
    row, col = get_row_col(alg)
    return np.array([row * SIDE - offset_x, col * SIDE - offset_y, 0])

#for i in np.arange(-3.5, 4.5, 1):
#    for j in np.arange(-3.5, 4.5, 1):
for row in 'abcdefgh':
    for col in range(1, 9):
        alg = f'{row}{col}'
        color = get_color(alg)
        r = get_center(alg)
        raw_squares.append(r)
        s = (square + r)
        p = project(s, FocalLength,
                    X, Y, Z, roll, pitch, yaw)
        if alg == 'a1':
            objects.append((ax.fill(p[:,0], p[:,1], alpha=.3, color='k')[0], color))
        elif color:
            objects.append((ax.fill(p[:,0], p[:,1], alpha=.3, color='r')[0], color))
        else:
            objects.append((ax.fill(p[:,0], p[:,1], alpha=.3, color='w')[0], color))

for row in 'ah':
    for col in [1, 8]:
        alg = f'{row}{col}'
        r = get_center(alg)
        p = project(r[np.newaxis], FocalLength,
                    X, Y, Z, roll, pitch, yaw)[0]
        #pl.plot(p[0], p[1], 'wd')
       

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
    for r, (o, color) in zip(raw_squares, objects):
        s = square + r
        p = project(s, FocalLength, X, Y, Z, roll, pitch, yaw)
        o.set_xy(p)
    
    
        
pl.axis('equal')

methods = ['Nelder-Mead',
           'Powell',    
           'CG', 
           'BFGS',    
           'L-BFGS-B', 
           'TNC',      
           'COBYLA',   
           'SLSQP',    
           ]

def onclick(event):
    global FocalLength, X, Y, Z, roll, pitch, yaw
    
    def minme(params, args, plot=False):
        print(params)
        if len(params) > 0:
            FocalLength = params[0]
        if len(params) > 3:
            X, Y, Z = params[1:4]
        if len(params) > 6:
            roll, pitch, yaw = params[4:7]
        else:
            roll, pitch, yaw = args
        err = 0
        for alg in positions:
            pixel_center = positions[alg]
            center = get_center(alg)
            projected_center = project(center[np.newaxis], FocalLength, X, Y, Z, roll, pitch, yaw)
            err += np.linalg.norm(projected_center - pixel_center)
            #err += prior_w @ (params - prior) ** 2
            if plot:
                print(alg, pixel_center, projected_center)
                pl.plot(projected_center[0,0], projected_center[0,1], 'gd')
                pl.plot(pixel_center[0], pixel_center[1], 'rd')
        return err

    for method in methods:
        guess= [FocalLength, X, Y, Z, roll, pitch, yaw ]
        guess = [FocalLength, X, Y, Z]
        args = [roll, pitch, yaw]
        ans = minimize(minme, guess ,method=method, args=args)['x']
        print(method, minme(ans, args, plot=True))
        fig.canvas.draw_idle()
        print(ans)
        break
    on_update()
    fig.canvas.draw_idle()
    return
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

cid = fig.canvas.mpl_connect('button_press_event', onclick)

pl.show()
### fit params
# FocalLength, X, Y, Z, roll, pitch, yaw

