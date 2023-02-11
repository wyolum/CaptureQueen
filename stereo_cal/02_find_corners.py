import cv2
import glob
import numpy as np

from constants import IM_WIDTH, IM_HEIGHT


images_folder = 'images/*'
images_names = sorted(glob.glob(images_folder))
images = []
for imname in images_names:
    im = cv2.imread(imname, 1)
    images.append(im)

#criteria used by checkerboard pattern detector.
#Change this if the code can't find the checkerboard
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
rows = 7 #number of checkerboard rows.
columns = 7 #number of checkerboard columns.
INCH = 25.4
world_scaling =  2.25 * INCH #change this to the real world square size. Or not.
world_scaling = 1

#coordinates of squares in the checkerboard world space
objp = np.zeros((rows*columns,3), np.float32)
objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
objp = world_scaling* objp

 
#frame dimensions. Frames should be the same size.
width = images[0].shape[1] / 2
height = images[0].shape[0]
assert width == IM_WIDTH

#coordinates of the checkerboard in checkerboard world space.
objpointss = [[],[]] # 3d point in real world space
 
 
#Pixel coordinates of checkerboards
cornerss = [[], []]
for i, frame in enumerate(images):
    grays = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    #find the checkerboard
    for k in range(2):
        gray = grays[:,k * IM_WIDTH:(k + 1) * IM_WIDTH]
        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)
        if ret == True:
            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)
            #opencv can attempt to improve the checkerboard coordinates
            corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cornerss[k].append(corners)
            objpointss[k].append(objp)
            cv2.drawChessboardCorners(frame[:,k * IM_WIDTH:(k + 1) * IM_WIDTH], (rows,columns), corners, ret)
            cv2.imshow('img', frame)
    print(f'{i}/{len(images)}')
    k = cv2.waitKey(-1)
npz = 'cornerss.npz'
np.savez(npz, cornerss=cornerss, objpointss=objpointss)
print(f'wrote {npz}')
