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

################################################################################
import math
from PIL import Image, ImageDraw
from PIL import ImagePath 
  
side = 6
xy = [
    ((math.cos(th) + 1) * 40,
     (math.sin(th) + 1) * 40)
    for th in [i * (2 * math.pi) / side for i in range(side)]
    ]

img = Image.new("RGBA", (80, 80), "#FFFFFF00") 
img1 = ImageDraw.Draw(img)  
img1.polygon(xy, fill ="#FFFFFFA0", outline ="blue")
images.append(ImageTk.PhotoImage(img))
canvas.create_image(200, 0, image=images[-1], anchor='nw')

################################################################################
root.mainloop()
