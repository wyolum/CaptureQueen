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

def create_polygon(xy, fill, outline, canvas):
    start = np.min(xy, axis=0)
    stop = np.max(xy, axis=0)
    xy -= start
    size = tuple((np.round(stop - start) + 1).astype(int))
    xy = [tuple(np.round(l)) for l in xy]

    img = Image.new("RGBA", size, "#FFFFFF00") 
    img1 = ImageDraw.Draw(img)
    img1.polygon(xy, fill=fill, outline=outline)
    images.append(ImageTk.PhotoImage(img))
    canvas.create_image(start[0], start[1], image=images[-1], anchor='nw')

#create_polygon(head, fill, outline, canvas)

def rotate_poly(xy, twist):
    rot = np.array([[ cos(twist), sin(twist)],
                    [-sin(twist), cos(twist)]])
    out = xy @ rot
    return out

def alg_to_center(alg):
    #[0, R] + square_size/2
    col = (ord(alg[0].upper()) - ord('A')) + .5
    row = 8.5 - int(alg[1])
    return np.array([col * square_size, row * square_size])

def arrow_to(from_alg, to_alg, fill, canvas, square_size):
    from_center = alg_to_center(from_alg)
    to_center = alg_to_center(to_alg)
    h = head + to_center
    print(from_center, to_center)
    theta = np.arctan2(to_center[0] - from_center[0],
                       to_center[1] - from_center[1])
    theta = pi - theta
    create_polygon(rotate_poly(head, theta) + to_center, fill, outline, canvas)

    l = np.linalg.norm(to_center - from_center) - R * (1 + sin(30 *DEG)) + 1
    w = R/3
    stem = np.array([[-w, 0],
                     [-w, -l],
                     [w, -l],
                     [w, 0]])
    print(theta / DEG)
    arrow = np.vstack([head + [0, -l], stem])
    create_polygon(rotate_poly(stem, theta) + from_center,
                   fill, outline, canvas)

if __name__ == '__main__':
    root = Tk()
    fill = "#404040A0"
    outline = "#00000000"
    DEG = pi / 180
    twist = 20 * DEG

    side = 3
    R = 30
    theta = np.arange(0, 2 * pi, 2 * pi / side) + pi
    head = np.column_stack([sin(theta), cos(theta)]) * R  + [0, R]
    print(head)

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
    arrow_to('B1', 'C3', fill, canvas, square_size)
    arrow_to('E2', 'E4', '#80000080', canvas, square_size)
    arrow_to('H1', 'A8', '#00800080', canvas, square_size)
    ############################################################################
    root.mainloop()
