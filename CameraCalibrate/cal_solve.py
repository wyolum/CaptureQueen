import glob
import cv2
import numpy as np
datadir = "images/"
images = glob.glob(datadir + '*.png')
images.sort()
images = images[::5]

from cv2 import aruco
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
board = aruco.CharucoBoard_create(7, 5, 1, .8, aruco_dict)

if False:
    import PIL.Image
    import matplotlib.pyplot as plt
    im = PIL.Image.open(images[0])
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.imshow(im)
    #ax.axis('off')
    plt.show()

def read_chessboards(images):
    """
    Charuco base pose estimation.
    """
    print("POSE ESTIMATION STARTS:")
    allCorners = []
    allIds = []
    decimator = 0
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

        if len(corners)>0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize = (3,3),
                                 zeroZone = (-1,-1),
                                 criteria = criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,gray,board)
            if (res2[1] is not None and res2[2] is not None and
                len(res2[1])>3 and decimator%1==0):
                allCorners.append(res2[1])
                allIds.append(res2[2])

        decimator+=1

    imsize = gray.shape
    return allCorners,allIds,imsize

def calibrate_camera(allCorners,allIds,imsize):
    """
    Calibrates the camera using the dected corners.
    """
    print("CAMERA CALIBRATION")

    cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
                                 [    0., 1000., imsize[1]/2.],
                                 [    0.,    0.,           1.]])

    distCoeffsInit = np.zeros((5,1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS +
             cv2.CALIB_RATIONAL_MODEL +
             cv2.CALIB_FIX_ASPECT_RATIO)
    #flags = (cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=allCorners,
                      charucoIds=allIds,
                      board=board,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT,
                                10000, 1e-9))

    return (ret, camera_matrix, distortion_coefficients0,
            rotation_vectors, translation_vectors)

allCorners, allIds, imsize = read_chessboards(images)
(ret, camera_matrix, distortion_coefficients0,
 rotation_vectors, translation_vectors) = calibrate_camera(
     allCorners,allIds,imsize)
print(ret)
saved = glob.glob('results/*.npz')
i = len(saved)
npz = f'results/{i:04d}.npz'
np.savez(npz, camera_matrix=camera_matrix,
         distorition_coeff=distortion_coefficients0,
         rotation_vectors=rotation_vectors,
         translation_vectors=translation_vectors)
print(distortion_coefficients0)
print(rotation_vectors)
print(translation_vectors)
print(f'saved {npz}')
