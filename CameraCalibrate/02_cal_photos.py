import cv2
from cv2 import aruco

vid = cv2.VideoCapture(0)
  ## 1920 x 1080
  ## 1024 x 768
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 786)
arucoDict = aruco.Dictionary_get(aruco.DICT_6X6_250)
arucoParam = aruco.DetectorParameters_create()

def findArucoMarkers(img, draw=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict,
                                               parameters = arucoParam)
    if ids is None:
        ids = []
    return ids

capture_i = 0
while capture_i < 100:
    ret, frame = vid.read()
    cv2.imshow('frame', frame)
    ids = findArucoMarkers(frame)
    if len(ids) == 17:
        png = f'images/{capture_i:04d}.png'
        cv2.imwrite(png, frame)
        print(png, len(ids))
        capture_i += 1
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break


