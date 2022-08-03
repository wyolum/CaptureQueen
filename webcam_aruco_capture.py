import cv2
import cv2.aruco as aruco


markerSize = 5
totalMarkers=100
key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
arucoDict = aruco.Dictionary_get(key)

def findArucoMarkers(img, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)
    _bboxs, _ids, _rejected = aruco.detectMarkers(255-gray, arucoDict, parameters = arucoParam)
    if draw:
        cv2.aruco.drawDetectedMarkers(img, bboxs)
        cv2.aruco.drawDetectedMarkers(img, _bboxs)
    if ids is not None:
        print(ids)
    if _ids is not None:
        print(_ids)
    return ids, _ids
  
# define a video capture object
vid = cv2.VideoCapture(0)
  ## 4056 x 3040
  ## 2592 × 1944
  ## 1920 x 1080
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 786)
#vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    #print(frame.shape)
    #break
  
    # Display the resulting frame
    findArucoMarkers(frame)
    cv2.imshow('frame', frame)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
