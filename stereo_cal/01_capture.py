import os
import os.path
import cv2
import glob

camera_number = 2
## setup

IM_WIDTH = 640 ### Single image
IM_HEIGHT = 480 ### 

vid = cv2.VideoCapture(camera_number)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 2 * IM_WIDTH)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, IM_HEIGHT)

N = 30

if not os.path.exists('stereo_cal_images'):
    os.mkdir('stereo_cal_images')

## 1.  Take N photos
n = len(glob.glob("images/[0-9][0-9][0-9].png"))
        

while n < N:
    ret, frame = vid.read()
    if ret:
        cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == 32:
        png = f'images/{n:03d}.png'
        cv2.imwrite(png, frame)
        print(png)
        n += 1
    if key == 113:
        break
