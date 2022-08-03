import numpy as np
import cv2, PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl

aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)
#aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)

fig = plt.figure(figsize=(8, 8))
nx = 8
ny = 4
s = 225
for i in range(0, 8):
    for j in range(0, 8):
        if (i + j) % 2 == 0:
            plt.fill([-s/2 + i * s,
                      -s/2 + i * s,
                      s/2 + i * s,
                      s/2 + i * s],
                     [-s/2 + j * s,
                      s/2 + j *  s,
                      s/2 + j * s,
                      -s/2 + j * s],
                     color='k')

def invert(imagem):
    return 255 - imagem

for i in range(8):
    for j in range(8):
        div = 5
        val = i * 8 + j
        extent = (-s/div + j * s, s/div + j * s,
                  -s/div + (7 - i) * s, s/div + (7 - i) * s)
        img = aruco.drawMarker(aruco_dict, val, s)
        if (i + j) % 2 == 0:
            pass
        else:
            img = invert(img)
        
        plt.imshow(img,
                   extent=extent,
                   cmap = mpl.cm.gray, interpolation = "nearest", zorder=100)
        #t = plt.text(j * s, (7 - i) * s, val, color='r', fontsize=24)
        #t.set_zorder(100)
color = 'black'
size = 14
for i in range(8):
    plt.text(-.75 * s, i * s, i+1,
             color=color, fontsize=size, va='center', ha='center')
    plt.text(7.75 * s, i * s, i+1,
             color=color, fontsize=size, va='center', ha='center', rotation=180)
    plt.text(i * s, -.75 * s, 'abcdefgh'[i],
             color=color, fontsize=size, va='center', ha='center')
    plt.text(i * s, 7.75 * s, 'abcdefgh'[i],
             color=color, fontsize=size, va='center', ha='center', rotation=180)
border = np.column_stack([[0, 0, 8, 8, 0],
                          [0, 8, 8, 0, 0]]) * s - s/2
Border = np.column_stack([[0, 0, 8, 8, 0],
                          [0, 8, 8, 0, 0]]) * s * 1.125 - s
plt.plot(border[:,0], border[:,1], 'k-', linewidth=2)
plt.plot(Border[:,0], Border[:,1], 'k-', linewidth=2)

plt.axis("off")
plt.xlim(-.5 * s, 7.75 * s)
plt.ylim(-.5 * s, 7.75 * s)
plt.savefig("markers.pdf")
print('wrote markers.pdf')
plt.show()
