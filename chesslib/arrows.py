from tkinter import *
from PIL import Image, ImageTk


images = []  # to hold the newly created image

def create_rectangle(x1, y1, x2, y2, **kwargs):
    if 'alpha' in kwargs:
        alpha = int(kwargs.pop('alpha') * 255)
        fill = kwargs.pop('fill')
        fill = root.winfo_rgb(fill) + (alpha,)
        image = Image.new('RGBA', (x2-x1, y2-y1), fill)
        images.append(ImageTk.PhotoImage(image))
        canvas.create_image(x1, y1, image=images[-1], anchor='nw')
    canvas.create_rectangle(x1, y1, x2, y2, **kwargs)

import numpy as np
from numpy import pi, cos, sin
from PIL import Image, ImageDraw
from PIL import ImagePath 

def create_polygon(xy, fill, outline, canvas, tags=None):
    start = np.min(xy, axis=0)
    stop = np.max(xy, axis=0)
    xy -= start
    size = tuple((np.round(stop - start) + 1).astype(int))
    xy = [tuple(np.round(l)) for l in xy]

    img = Image.new("RGBA", size, "#FFFFFF00") 
    img1 = ImageDraw.Draw(img)
    img1.polygon(xy, fill=fill, outline=outline)
    images.append(ImageTk.PhotoImage(img))
    if tags is None:
        tags = ['polygon']
    else:
        tags = ['polygon'] + list(tags)
    canvas.create_image(start[0], start[1], image=images[-1], anchor='nw',
                        tags=tags)

#create_polygon(head, fill, outline, canvas)

def rotate_poly(xy, twist):
    rot = np.array([[ cos(twist), sin(twist)],
                    [-sin(twist), cos(twist)]])
    out = xy @ rot
    return out

def alg_to_center(alg, square_size):
    #[0, R] + square_size/2
    col = (ord(alg[0].upper()) - ord('A')) + .5
    row = 8.5 - int(alg[1])
    return np.array([col * square_size, row * square_size])

UNIT_SQUARE = np.array([[ 1,  1],
                        [ 1, -1],
                        [-1, -1],
                        [-1,  1]]) / 2

def color_square(alg, fill, outline, canvas, square_size):
    center = alg_to_center(alg, square_size)
    square = UNIT_SQUARE * square_size + center
    tags = ['square']
    create_polygon(square, fill, outline, canvas, tags=tags)

theta = np.linspace(0, pi, 100)
semicircle = np.column_stack([cos(theta), sin(theta)]) - [0, 1]
def arrow_to(from_alg, to_alg, fill, outline, canvas, square_size):
    from_center = alg_to_center(from_alg, square_size)
    to_center = alg_to_center(to_alg, square_size)
    h = head + to_center
    theta = np.arctan2(to_center[0] - from_center[0],
                       to_center[1] - from_center[1])
    theta = pi - theta
    R = square_size * .53
    
    l = np.linalg.norm(to_center - from_center) - R * (1 + sin(pi/6)) + 1
    w = R/3
    stem = np.array([[-w, -w],
                     [-w, -l],
                     [w, -l],
                     [w, -w]])
    stem = np.vstack([stem, semicircle * w])
    arrow = np.vstack([head * R + [0, -l], stem])
    xtra = R * np.sqrt(2)
    tags = ['arrow']
    create_polygon(rotate_poly(head * R, theta) + to_center, fill,
                   outline, canvas, tags=tags)    
    create_polygon(rotate_poly(stem, theta) + from_center,
                   fill, outline, canvas, tags=tags)

theta = np.arange(0, 2 * pi, 2 * pi / 3) + pi
R = 30
head = np.column_stack([sin(theta), cos(theta)]) + [0, 1]
if __name__ == '__main__':
    root = Tk()
    fill = "#404040A0"
    outline = "#00000000"
    DEG = pi / 180
    twist = 20 * DEG

    R = 30

    square_size = 56
    canvas = Canvas(width=8 * square_size, height=8 * square_size)
    canvas.pack()

    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                color = 'lightgrey'
            else:
                color = 'darkgrey'
            create_rectangle(i * square_size, j * square_size,
                             (i + 1) * square_size, (j + 1) * square_size,
                             fill=color)
    arrow_to('B1', 'C3', fill, outline, canvas, square_size)
    arrow_to('E2', 'E4', '#80000080', outline, canvas, square_size)
    arrow_to('H1', 'A8', '#00800080', outline, canvas, square_size)
    ############################################################################
    root.mainloop()
