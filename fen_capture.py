import time
import re
import numpy as np
import cv2
import cv2.aruco as aruco
import chess
import grid_perspective

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
order = []
ii = 0
for row in range(8):
    for col in range(8):
        letter = chr(col + ord('a'))
        number = row + 1
        val = (7 - row) * 8 + col
        alg = f'{letter}{number}'
        board_map[val] = [alg, None, None] # alg, box, center
        alg_map[alg] = val
        order.append((alg, ii))
        # print(alg, ii)
        ii += 1

def findArucoMarkers(img, draw=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict,
                                               parameters = arucoParam)
    _bboxs, _ids, _rejected = aruco.detectMarkers(255-gray, arucoDict,
                                                  parameters = arucoParam)
    if draw:
        cv2.aruco.drawDetectedMarkers(img, bboxs)
        cv2.aruco.drawDetectedMarkers(img, _bboxs)
        if False:
            if ids is not None and len(ids) > 0:
                for bbox, id in zip(bboxs, ids):
                    center = np.mean(bbox, axis=-2)[0]
                    #cv2.circle(img, tuple(center), 2, (0, 0, 254), 2)
                    #font = cv2.FONT_HERSHEY_SIMPLEX
                    #cv2.putText(img, str(id), tuple(center), font, 1 ,
                    #            (255, 255, 0))

            if _ids is not None and len(_ids) > 0:
                for bbox, id in zip(_bboxs, _ids):
                    center = np.mean(bbox, axis=-2)[0]
                    #cv2.circle(img, tuple(center), 2, (0, 254, 0), 2)
                    #font = cv2.FONT_HERSHEY_SIMPLEX
                    #cv2.putText(img, str(id), tuple(center), font, 1 ,
                    #            (255, 255, 0))
                
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
            last_pos = board_map[id][2]
            pos = np.mean(bbox[0], axis=0)
            if last_pos is not None:
                d = np.linalg.norm(last_pos - pos)
                if d > 2: ### board moved! clear out last known positions
                    print(id, 'board moved!')
                    for i in range(64):
                        board_map[i][1] = None
                        board_map[i][2] = None
            board_map[id][1] = bbox[0]
            board_map[id][2] = pos
                    
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

        #cv2.polylines(frame, bbox, True, (0, 0, 255), 1)
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

DEG = np.pi / 180
guess = [-100, -300, 10000, 12 * DEG, 0 * DEG, 90 * DEG]
def linfit(centers):
    ## try y = mx + b
    x, y = centers.T
    n = len(centers)
    A = np.column_stack([x, np.ones(n)])
    return np.linalg.inv(A.T @ A) @ (A.T @ y)

def find_intersection(lines):
    ### find intersection of two lines
    ### a0x + b0 = a1x + b1
    ### x(a0 - a1) = b1 - b0
    ### x = (b1 - b0) / (a0 - a1)
    lines = np.array(lines)
    a0, a1 = lines[:, 0]
    b0, b1 = lines[:, 1]
    x = (b1 - b0) / (a0 - a1)
    y = a0 * x + b0
    return np.array([x, y])

def average_intersection(lines):
    lines = np.array(lines)
    n = len(lines)
    pts = []
    for i in range(n):
        for j in range(i + 1, n):
            pts.append(find_intersection(lines[[i, j]]))
    return np.mean(pts, axis=0)

def draw_perspective(img, ids, centers):
    ### find vanishing point for columns
    algs = [board_map[id][0] for id in ids]
    coords = np.array([val_to_coords(id) for id in ids])
    cols = coords[:,0]
    rows = coords[:,1]
    row_lines = []
    for r in set(rows):
        keep = rows == r
        line = centers[keep]
        if len(line) > 1:
            a, b = linfit(centers[keep])
            row_lines.append([a, b])
            x = np.array([-1000, 1000])
            y = (a * x + b).astype(int)
            cv2.line(img, (x[0], y[0]), (x[1], y[1]), (0, 255, 123), 2)
    vanish = average_intersection(row_lines).astype(int)
    cv2.circle(img, tuple(vanish), 4, (56, 123, 26), 4)
    col_lines = []
    for c in set(cols):
        keep = cols == c
        line = centers[keep]
        if len(line) > 1:
            a, b = linfit(centers[keep])
            col_lines.append([a, b])
            x = np.array([-1000, 1000])
            y = (a * x + b).astype(int)
            cv2.line(img, (x[0], y[0]), (x[1], y[1]), (255, 123, 0), 2) 
    vanish = average_intersection(col_lines).astype(int)
    cv2.circle(img, tuple(vanish), 4, (56, 123, 26), 4)
            
while(True):
    # Capture the video frame
    ret, frame = vid.read()
    bboxs, free_ids = findArucoMarkers(frame, draw=True)
    centers = np.mean(bboxs, axis=-2).squeeze()
    focal_length = 1000
    side = 1000
    virtual_board = grid_perspective.ChessBoard(side, 'g', 'w')
    centers = []
    ids = []
    for id in range(64):
        bbox = board_map[id][1]
        if bbox is not None:
            center = np.mean(bbox, axis=0)
            centers.append(center)
            ids.append(id)
    ids = np.array(ids)
    centers = np.array(centers)
    guess = draw_perspective(frame, ids, centers)
    
        
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
    cv2.imshow('frame', frame)# [::-1,::-1])
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
 
