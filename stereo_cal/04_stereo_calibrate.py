# https://learnopencv.com/making-a-low-cost-stereo-camera-using-opencv/
import glob
import numpy as np
import cv2

from constants import IM_WIDTH, IM_HEIGHT

images_folder = 'images/*'
imname = sorted(glob.glob(images_folder))[0]
im = cv2.imread(imname, 1)
width = im.shape[1] // 2
height = im.shape[0]

imgL = im[:,0 * IM_WIDTH:(0 + 1) * IM_WIDTH]
imgR = im[:,1 * IM_WIDTH:(1 + 1) * IM_WIDTH]


npy = 'calz.npy'
calz = np.load(npy, allow_pickle=True)
pts = np.load('cornerss.npz')
obj_pts = pts['objpointss'][0]

img_ptsL = pts['cornerss'][0]
img_ptsR = pts['cornerss'][1]

mtxL  = calz[0]['mtx']
distL = calz[0]['dist']
mtxR  = calz[1]['mtx']
distR = calz[1]['dist']

flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same 
 
criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
 
# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix

imageSize = (IM_WIDTH, IM_HEIGHT)
result = cv2.stereoCalibrate(obj_pts, img_ptsL, img_ptsR, mtxL, distL, mtxR, distR, imageSize, criteria_stereo, flags)
#result = cv2.stereoCalibrate(obj_pts, img_ptsL, img_ptsR, imageSize, None, None, None, None)

retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat= result

OPTIMIZE_ALPHA = 0.25
result = cv2.stereoRectify(new_mtxL, distL,
                           new_mtxR, distR,
                           imageSize, Rot, Trns,
                           None, None, None, None, None,
                           cv2.CALIB_ZERO_DISPARITY, OPTIMIZE_ALPHA)

(leftRectification, rightRectification, leftProjection, rightProjection,
        dispartityToDepthMap, leftROI, rightROI) = result

leftMapX, leftMapY = cv2.initUndistortRectifyMap(
    new_mtxL, distL, leftRectification,
    leftProjection, imageSize, cv2.CV_32FC1)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(
        new_mtxR, distR, rightRectification,
        rightProjection, imageSize, cv2.CV_32FC1)

npz = 'stereo_calz.npz'
np.savez_compressed(npz, imageSize=imageSize,
                    leftMapX=leftMapX, leftMapY=leftMapY, leftROI=leftROI,
                    rightMapX=rightMapX, rightMapY=rightMapY, rightROI=rightROI)

print(f'wrote {npz}')
