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
stereoMatcher.setMinDisparity(4)
stereoMatcher.setNumDisparities(128)
stereoMatcher.setSpeckleRange(16)
stereoMatcher.setSpeckleWindowSize(45)
stereoMatcher.setBlockSize(15)


REMAP_INTERPOLATION = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4][4]

still_image = cv2.imread('image.jpg')
im = still_image
def get_stereo_image():
    #ret, im = vid.read()
    #cv2.imwrite('image.jpg', im)
    return still_image
    images_folder = 'images/*'
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv2.imread(imname, 1)
        images.append(im)
        return im


def match_block_x(block, img):
    n, m = block.shape
    block = block.ravel()
    block = block / np.linalg.norm(block)
    out = []
    for j in range(img.shape[1] - m):
        oth = img[:,j:j+m].ravel()
        norm = np.linalg.norm(oth)
        if norm > 0:
            oth = oth / norm
        out.append(oth @ block)
    return np.array(out)

im = get_stereo_image()

imgL = im[:,0 * IM_WIDTH:(0 + 1) * IM_WIDTH]
imgR = im[:,1 * IM_WIDTH:(1 + 1) * IM_WIDTH]
    
fixedLeft = cv2.remap(imgL, leftMapX, leftMapY, REMAP_INTERPOLATION)
fixedRight = cv2.remap(imgR, rightMapX, rightMapY, REMAP_INTERPOLATION)
grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
grayLeft = fixedLeft[:,:,0]
grayRight = fixedRight[:,:,0]

fig, ax = pl.subplots(1, 3, num=1, clear=True, figsize=(12, 4), sharex=True, sharey=True)
ax[0].imshow(grayLeft, cmap='gray')

block_size= (5, 100)
n, m = grayLeft.shape

img_blur = cv2.GaussianBlur(grayRight,(3,3), sigmaX=0, sigmaY=0) 
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) 
 

for i in range(450, n):
    row = grayLeft[i]
    for j in range(800, m):
        block = grayLeft[i:i+block_size[1],j:j+block_size[0]]
        #res = cv2.matchTemplate(grayRight[i:i+block_size[0],j:j+block_size[0] + 100], block, cv2.TM_CCOEFF_NORMED)
        #res = match_block_x(block, grayRight[i:i+block_size[1],j-100:j+block_size[0] + 100])
        res = match_block_x(block, grayRight[i:i+block_size[1], j-100:j+100])
        pl.figure()
        pl.plot(res)
        ax[2].plot(res)
        mx = np.max(res)
        x = np.where(res == mx)
        x = j + x[0]/2
        y = i
        ax[1].imshow(grayRight, cmap='gray')
        ax[2].imshow(block, cmap='gray')
        ax[1].plot(x, y, 'r.')
        ax[0].plot([j, j, j + block_size[0], j + block_size[0], j],
                   [i, i + block_size[1], i + block_size[1], i, i], 'r-')
        ax[1].plot([j, j, j + block_size[0], j + block_size[0], j],
                   [i, i + block_size[1], i + block_size[1], i, i], 'g-')
        ax[1].plot([x, x, x + block_size[0], x + block_size[0], x],
                   [y, y + block_size[1], y + block_size[1], y, y], 'r-')
        break
        ax[1].imshow(grayRight[x:x+block_size[0], y:y+block_size[1]], cmap='gray')
        ax[2].plot(0, 0, 'r.')
        break
    break
pl.show()
