import numpy as np
import glob
import cv2

images_folder = 'images/*.png'
imname = sorted(glob.glob(images_folder))[0]
im = cv2.imread(imname, 1)
width = im.shape[1] // 2
height = im.shape[0]

d = np.load('cornerss.npz')


calz = []
mtxs = []
dists = []
for k in range(2):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(d['objpointss'][k], d['cornerss'][k], (width, height), None, None)
    print(d['objpointss'][k])
    calz.append({'mtx':mtx, 'dist':dist})
    mtxs.append(mtx)
    dists.append(dist)
    #print(k)
    #print(ret)
    #print(mtx)
    #print(dist)
    #print(rvecs)
    #print(tvecs)
npy = 'calz.npy'
np.save(npy, calz)
d = np.load(npy, allow_pickle=True)
print(f'Wrote {npy}')
for k in range(2):
    assert np.linalg.norm(d[k]['mtx'] - mtxs[k]) < 1e-8
    assert np.linalg.norm(d[k]['dist'] - dists[k]) < 1e-8
