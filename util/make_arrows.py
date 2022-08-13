import numpy as np
from PIL import Image, ImageDraw

from numpy import sin, cos, pi

square_size=56

head_size = int(square_size/2)

head = Image.new('RGBA', (square_size, square_size))
head1 = ImageDraw.Draw(head)
points = (head_size * np.array([[0, 0],
                                [cos(pi/4), sin(pi/4)],
                                [cos(3 * pi / 4), sin(3 * pi / 4)]])).astype(int) + 100
print(points)
head1.polygon(points, fill ="black", outline ="black")
head.show()
from tkinter import *
from PIL import Image, ImageTk

root = Tk()

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

canvas = Canvas(width=300, height=200)
canvas.pack()

create_rectangle(10, 10, 200, 100, fill='blue')
create_rectangle(50, 50, 250, 150, fill='green', alpha=.5)
create_rectangle(80, 80, 150, 120, fill='#800000', alpha=.8)

images.append(ImageTk.PhotoImage(head))
canvas.create_image(100, 100, image=images[-1], anchor='nw')

root.mainloop()
