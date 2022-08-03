import time
import re
import numpy as np
import cv2
import cv2.aruco as aruco
import chess

markerSize = 5
totalMarkers=100
key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
arucoDict = aruco.Dictionary_get(key)
arucoParam = aruco.DetectorParameters_create()

#arucoParam.minDistanceToBorder = 2
#arucoParam.adaptiveThreshWinSizeMax = 40
#for x in dir(arucoParam):
#    print(x, getattr(arucoParam, x))
board_map = {}
alg_map = {}
for row in range(8):
    for col in range(8):
        letter = chr(col + ord('a'))
        number = row + 1
        val = (7 - row) * 8 + col
        alg = f'{letter}{number}'
        board_map[val] = [alg, None]
        alg_map[alg] = val

def findArucoMarkers(img, draw=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict,
                                               parameters = arucoParam)
    _bboxs, _ids, _rejected = aruco.detectMarkers(255-gray, arucoDict,
                                                  parameters = arucoParam)
    if draw:
        cv2.aruco.drawDetectedMarkers(img, bboxs)
        cv2.aruco.drawDetectedMarkers(img, _bboxs)
        #cv2.aruco.drawDetectedMarkers(img, rejected)
        #cv2.aruco.drawDetectedMarkers(img, _rejected)

    out_bboxs = []
    out_ids = []
    if ids is not None:
        out_ids.extend(ids)
        out_bboxs.extend(bboxs)
    if _ids is not None:
        out_ids.extend(_ids)
        out_bboxs.extend(_bboxs)

    #out = np.array(out).ravel()
    out_ids = np.array(out_ids).ravel()
    for id, bbox in zip(out_ids, out_bboxs):
        if 0 <= id and id < 64:
            board_map[id][1] = bbox[0]
    return out_bboxs, out_ids
        
def alg_dist(alg0, alg1):
    dx = ord(alg0[0]) - ord(alg1[0])
    dy = ord(alg0[1]) - ord(alg1[1])
    return np.sqrt(dx ** 2 + dy ** 2), dx, dy

def crop_square(frame, alg):
    ### find closes mapped square
    ids = board_map.keys()
    mindist = np.inf
    for i, id in enumerate(ids):
        bbox = board_map[id][1]
        if bbox is not None:
            d, dx, dy = alg_dist(alg, board_map[id][0])
            if d < mindist:
                mindist = d
                mini = i
                mindx = dx
                mindy = dy
                minbbox = bbox
                if d == 0:
                    break
    if mindist < 20:
        bbox = minbbox
        center = np.mean(bbox, axis=0)
        ihat = bbox[1] - bbox[0]
        sx = np.linalg.norm(ihat)
        jhat = bbox[1] - bbox[2]
        sy = np.linalg.norm(jhat)
        scale = 2.5
        ihat *= scale / sx
        jhat *= scale / sy
        dx = mindx
        dy = mindy

        bbox = (bbox - center) * scale + center
        bbox += dx * sx * ihat + dy * sy * jhat 
        starts = np.min(bbox, axis=0).astype(int)
        stops = np.max(bbox, axis=0).astype(int) +1
        bbox = bbox.reshape((1, -1, 1, 2)).astype(np.int32)

        cv2.polylines(frame, bbox, True, (0, 0, 255), 1)
        return frame[starts[1]:stops[1],starts[0]:stops[0]]
        
def val_to_coords(val):
    row, col = divmod(val, 8)
    return (col, 7 - row)

board = chess.Board()
#fen = open('.fen').read().strip()
#fen = '4r3/1p3pk1/1bp1r1p1/p2qP2p/2R5/1P3N1P/5PP1/2Q1R1K1 w - - 1 33'
#board = chess.Board(fen)
open('.fen', 'w').write(board.fen())
print(board)

def get_occupied_squares(board):
    s = str(board).replace('\n', '').replace(' ', '')
    p_x = re.compile('[rpnbqkRPNBQK]')
    s = re.sub('[rpnbqkRPNBQK]', 'X', s)
    return s

def average_square(data):
    h, w, d = data.shape
    x = w//5
    y = h//5
    
    return np.mean(data[y:-y,x:-x])

def find_move(frame, free):
    start = time.time()

    coords = []
    occupied = ['X' for i in range(64)]
    
    for val in free:
        if 0 <= val and val < 64:
            occupied[val] = '.'
    occupied = ''.join(occupied)
    print(get_occupied_squares(board))
    print(occupied)
    print()
    candidates = []
    for move in board.legal_moves:
        board.push_uci(move.uci())
        test = get_occupied_squares(board)
        if occupied == test:
            candidates.append(move)
        board.pop()

    out = None
    if len(candidates) == 1:
        out = candidates[0]
    if len(candidates) > 1:
        ## get squres in question
        s0 = candidates[0].uci()[2:]
        s1 = candidates[1].uci()[2:]
        s0_val = average_square(crop_square(frame, s0))
        s1_val = average_square(crop_square(frame, s1))
        if board.turn == chess.WHITE: ### choose higher average value
            if s0_val > s1_val:
                out = candidates[0]
            else:
                out = candidates[1]
        else: ### choose smaller average value
            if s0_val < s1_val:
                out = candidates[0]
            else:
                out = candidates[1]
    if out:
        board.push_uci(out.uci())
        print(board.fen())
        print(board.fen(), file=open(".fen", 'w'), flush=True)
        
        
    return out

vid = cv2.VideoCapture(0)
  ## 1920 x 1080
  ## 1024 x 768
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 786)

move_number = 1
while(True):
    # Capture the video frame
    ret, frame = vid.read()
    bboxs, free_ids = findArucoMarkers(frame, draw=True)
    if True:
        for letter in 'abcdefgh':
            for number in range(1, 9):
                alg = f'{letter}{number}'
                png = f'img/{alg}.png'
                square = crop_square(frame, alg)
                #cv2.imwrite(png, square)
                #print(png)
        #here
        
    if True:
        if len(board.move_stack) > 0:
            alg = board.move_stack[-1].uci()
            crop_square(frame, alg[:2])
            crop_square(frame, alg[2:])
    cv2.imshow('frame', frame[::-1,::-1])
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('x'):
        print('clock')
        png = f'captures/{move_number:04d}.png'
        move_number += 1
        cv2.imwrite(png, frame)
        move = find_move(frame, free_ids)
        if move:
            open('.fen', 'w').write(board.fen())
    
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
 
