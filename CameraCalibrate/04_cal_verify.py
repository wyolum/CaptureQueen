import glob
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.interactive(True)
import cv2

datadir = "images/"
images = glob.glob(datadir + '*.png')
images.sort()
images = images[::5]

npzs = glob.glob('results/*.npz')
npzs.sort()
npz = npzs[-1]
results = np.load(npz)
camera_matrix = results['camera_matrix']
distorition_coeff = results['distorition_coeff']

images = glob.glob(datadir + '*.png')

import time
figure = plt.figure(1, figsize=(12, 6))
fig, ax = plt.subplots(1, 2, num=1)
for i in range(len(images)):
    frame = cv2.imread(images[i])
    start = time.time()
    img_undist = cv2.undistort(frame,camera_matrix,distorition_coeff,None)
    ax[0].imshow(frame)
    ax[0].set_title("Raw image")
    ax[0].axis("off")

    ax[1].imshow(img_undist)
    ax[1].set_title("Corrected image")
    ax[1].axis("off")
    input('...')
    #plt.show()
