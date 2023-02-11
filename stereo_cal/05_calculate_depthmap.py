# https://albertarmea.com/post/opencv-stereo-camera/
import pylab as pl
import cv2
import glob
import numpy as np

from constants import IM_WIDTH, IM_HEIGHT

npz = 'stereo_calz.npz'

calibration = np.load(npz, allow_pickle=False)
imageSize = tuple(calibration["imageSize"])

leftMapX = calibration["leftMapX"]
leftMapY = calibration["leftMapY"]
leftROI = tuple(calibration["leftROI"])

rightMapX = calibration["rightMapX"]
rightMapY = calibration["rightMapY"]
rightROI = tuple(calibration["rightROI"])

stereoMatcher = cv2.StereoBM_create()
#stereoMatcher.setMinDisparity(4)
#stereoMatcher.setNumDisparities(128)
#stereoMatcher.setSpeckleRange(16)
#stereoMatcher.setSpeckleWindowSize(5)
stereoMatcher.setBlockSize(31)


camera_number = 2

vid = cv2.VideoCapture(camera_number)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 2 * IM_WIDTH)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, IM_HEIGHT)

REMAP_INTERPOLATION = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4][4]

still_image = cv2.imread('image.jpg')
im = still_image
def get_stereo_image():
    #cv2.imwrite('image.jpg', im)
    #return still_image
    ret, im = vid.read()
    return im
    images_folder = 'images/*'
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv2.imread(imname, 1)
        images.append(im)
        return im


while 1:
    im = get_stereo_image()
    #cv2.imshow("im", im)
    key = cv2.waitKey(1)
    if key == 113:
        break
    
    imgL = im[:,0 * IM_WIDTH:(0 + 1) * IM_WIDTH]
    imgR = im[:,1 * IM_WIDTH:(1 + 1) * IM_WIDTH]
    
    fixedLeft = cv2.remap(imgL, leftMapX, leftMapY, REMAP_INTERPOLATION)
    fixedRight = cv2.remap(imgR, rightMapX, rightMapY, REMAP_INTERPOLATION)
    #fixedLeft = imgL
    #fixedRight = imgR
    grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
    
    depth = stereoMatcher.compute(grayLeft, grayRight)

    cv2.imshow('imgL', imgL)
    cv2.imshow('fixedLeft', fixedLeft)
    cv2.imshow('grayLeft', grayLeft)
    #cv2.imshow('depth', (depth - np.min(depth)) * (np.max(depth) - np.min(depth)))
    cv2.imshow('depth', depth)
