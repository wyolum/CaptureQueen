import sys
import argparse
import threading
import os.path
import time
import re
import numpy as np
import cv2

import chess
from board_map import fit, predict
from pygame_render import PygameRender
from mqtt_clock_client import mqtt_subscribe, mqtt_start, mqtt_clock_reset
from mqtt_clock_client import mqtt_publish_fen

initial_seconds = 300
initial_increment = 0

mqtt_pending_msgs = []
def on_mqtt(msg):
    mqtt_pending_msgs.append(msg)

def mqtt_handle_events():
    out = {}
    while mqtt_pending_msgs:
        msg = mqtt_pending_msgs.pop()
        if msg.topic not in out:
            out[msg.topic] = []
        out[msg.topic].append(msg.payload)
    if out:
        print(out)
    return out

mqtt_subscribe(on_mqtt)
mqtt_start()


desc = 'Capture Queen: Over-the-board real-time chess capture system.'
shortcuts = '''\
During game play, these keys are active:
    's' to swap colors
    'f' to flip sides
    'r' to reset to new game
    'q' to quit
    'x' to make move
'''
    
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-c','--calibrate',
                    help='Calibrate board area',
                    required=False, default=False)
parser.add_argument('-s','--shortcuts',
                    help='get gameplay command keys',
                    action="store_true")
args = parser.parse_args()
if args.shortcuts:
    print(shortcuts)
    sys.exit()

board_map = {}
alg_map = {}
order = []
ii = 0

hori_vanish_id = -1
vert_vanish_id = -2
righ_vanish_id = -3
left_vanish_id = -4

board_map[hori_vanish_id] = ['h-', None, None]
board_map[vert_vanish_id] = ['v-', None, None]
board_map[left_vanish_id] = ['l-', None, None]
board_map[righ_vanish_id] = ['r-', None, None]

alg_map['h-'] = hori_vanish_id
alg_map['v-'] = vert_vanish_id
alg_map['l-'] = left_vanish_id
alg_map['r-'] = righ_vanish_id

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
                if d > 10: ### board moved! clear out last known positions
                    print(id, 'board moved!')
                    for i in board_map:
                        board_map[i][1] = None
                        board_map[i][2] = None
            board_map[id][1] = bbox[0]
            board_map[id][2] = pos
                    
    return out_bboxs, out_ids
        
def alg_dist(alg0, alg1):
    dx = ord(alg0[0]) - ord(alg1[0])
    dy = ord(alg0[1]) - ord(alg1[1])
    return np.sqrt(dx ** 2 + dy ** 2), dx, dy

def get_board_bbox():
    delta = IM_HEIGHT / 9
    bbox = (np.array([
        [-0.5,  7.5],
        [-0.5, -0.5],
        [ 7.5, -0.5],
        [ 7.5,  7.5],
    ]) + 1) * delta
    return bbox

def get_abs_bbox(i, j):
    delta = IM_HEIGHT / 9
    coords = np.array([[-1, -1],
                       [-1,  1],
                       [ 1,  1],
                       [ 1, -1]]) / 6 + np.array([i, j])
    bbox = coords * delta
    return bbox.astype(int)

### flip_board
### If True, the right side of board from camera view point is toward the
### bottom of the board.
### If False, the left side of the board is toward the bottom

flip_board = False 
side = chess.WHITE
def get_bbox(alg):
    i = ord(alg[0]) - ord('a') + 1
    j = 9 - int(alg[1])
    if side == chess.BLACK:
        j = 9 - j
        i = 9 - i
    bbox = get_abs_bbox(i, j)
    return bbox

def draw_abs_square(rect, i, j, color, thickness):
    bbox = get_abs_bbox(i, j)
    cv2.rectangle(rect, tuple(bbox[0]), tuple(bbox[2]), color, thickness)
    
def draw_square(rect, alg, color, thickness):
    bbox = get_bbox(alg)
    cv2.rectangle(rect, tuple(bbox[0]), tuple(bbox[2]), color, thickness)

def crop_abs_square(rect, i, j):
    bbox = get_abs_bbox(i, j)
    starts = np.min(bbox, axis=0).astype(int)
    stops = np.max(bbox, axis=0).astype(int) + 1
    bbox = bbox.reshape((1, -1, 1, 2)).astype(np.int32)
    out = rect[starts[1]:stops[1],starts[0]:stops[0]], bbox
    return out
    
def crop_square(rect, alg):
    bbox = get_bbox(alg)
    starts = np.min(bbox, axis=0).astype(int)
    stops = np.max(bbox, axis=0).astype(int) + 1
    bbox = bbox.reshape((1, -1, 1, 2)).astype(np.int32)
    out = rect[starts[1]:stops[1],starts[0]:stops[0]], bbox
    return out
        
def val_to_coords(val):
    row, col = divmod(val, 8)
    return (col, 7 - row)

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
PURPLE = (255, 0, 355)
WHITE = (255, 255, 255)
BLACK = (0, 0 ,0)
GRAY = (128, 128, 128)

board = chess.Board()
#fen = open('.fen').read().strip()
## castle test
#fen = 'rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4'
## en passant test
#fen = 'rnbqkbnr/ppppp1p1/8/4Pp1p/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3'
## promotion test
#fen = 'rnb2rk1/ppppPpp1/8/8/7p/8/PPPP1PPP/RNB1KBNR w KQ - 0 8'
#board = chess.Board(fen)
open('.fen', 'w').write(board.fen())
#print(board)

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

last_rect = None
def find_move(rect):
    global last_rect
    if last_rect is not None:
        delta = cv2.absdiff(rect, last_rect)
        sum_delta = np.sum(delta)
        BUMP_THRESH = 5000000
        if sum_delta > BUMP_THRESH:
            print('Board moved {sum_delta} > {BUMP_THRESH}')
        imgray = cv2.cvtColor(delta,cv2.COLOR_BGR2GRAY)
        imgreen = delta[:,:,1]
        imred = delta[:,:,0]
        imblue = delta[:,:,2]
        thresh = np.max(delta, axis=-1)
        
        #ret,thresh = cv2.threshold(imgreen,25,255,0)
        #thresh = np.any(np.where(delta < 10, False, True), axis=2)  * 255
            
    else:
        thresh = None
    last_rect = rect.copy()

    changes = []
    if thresh is None:
        return

    candidates = []
    for move in board.legal_moves:
        ## only allow queen promotion at this time
        if move.promotion and move.promotion != chess.QUEEN:
            continue
        uci = move.uci()
        sq0, bbox0 = crop_square(thresh, uci[:2])
        sq1, bbox1 = crop_square(thresh, uci[2:4])
        sqs = [sq0, sq1]
        if board.is_castling(move):
            if uci[2] == 'g': ## kingside
                row = uci[3]
                sq2, bbox2 = crop_square(thresh, f'h{row}')
                sq3, bbox3 = crop_square(thresh, f'f{row}')
                sqs.extend([sq2, sq3])
            if uci[2] == 'c': ## queen
                row = uci[3]
                sq2, bbox2 = crop_square(thresh, f'a{row}')
                sq3, bbox3 = crop_square(thresh, f'd{row}')
                sqs.extend([sq2, sq3])
        if board.is_en_passant(move):
            col = uci[2]
            row = uci[1]
            sq2, bbox2 = crop_square(thresh, f'{col}{row}')
            sqs.append(sq2)
            
        change = np.array([int(np.sum(sq)) for sq in sqs])
        total_change = np.sum(change)
        change_thresh = 5000
        change_count = np.sum(change > change_thresh)
        #print(uci, move.from_square, move.to_square, move.promotion,
        #      board.is_en_passant(move),
        #      board.is_castling(move), change_count)
        if change_count == len(change):
            candidates.append([move, change_count, total_change])
    if len(candidates) == 0:
        out = None
    if len(candidates) == 1:
        out = candidates[0][0].uci()
    if len(candidates) > 1:
        sorted = np.argsort([c[2] for c in candidates])
        candidates = [candidates[i] for i in sorted]
        out = candidates[-1][0].uci()
    if out is not None:
        delta = IM_HEIGHT / 8
        box = np.array([[-1, -1],
                        [-1,  1],
                        [ 1,  1],
                        [ 1, -1.]]) * .5 * delta
        for u in range(2):
            i, j = val_to_coords(alg_map[out[2 * u:2 * (u + 1)]])
            bbox = box + np.int32([(i + .5) * delta, (8 - j - .5) * delta])
            bbox = bbox.reshape((1, -1, 1, 2)).astype(np.int32)

        
        draw_square(thresh, out[0:2], WHITE, 1)
        draw_square(thresh, out[2:4], WHITE, 1)
        
        board.push_uci(out)
        print(board.fen())
        print(board.fen(), file=open(".fen", 'w'), flush=True)
        
        
    return out

vid = cv2.VideoCapture(0)
  ## 1920 x 1080
  ## 1024 x 768
IM_WIDTH = 1024
IM_HEIGHT = 786
IM_WIDTH = 1920
IM_HEIGHT = 1080
IM_WIDTH = 640
IM_HEIGHT = 480
vid.set(cv2.CAP_PROP_FRAME_WIDTH, IM_WIDTH)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, IM_HEIGHT)

move_number = 1

DEG = np.pi / 180
grid_guess = [-27214, -3834, 43305, 2.2340, 0.8692, -1.0829]

def linfit(centers):
    ## try y = mx + b
    x, y = centers.T
    n = len(centers)
    A = np.column_stack([x, np.ones(n)])
    return np.linalg.pinv(A.T @ A) @ (A.T @ y)

def linfit2d(ij, x):
    '''
    x ~= i * a + j * b + c
    '''
    n = len(x)
    A = np.column_stack([ij, np.ones(n)])
    return np.linalg.pinv(A.T @ A) @ (A.T @ x)

def expfit(centers):
    ## y = a * e^(bx)
    ## logy = log(a) + b log(x)
    loga, b = linfit(np.log(centers))
    a = np.exp(loga)
    return np.array([a, b])

def quadfit2d(ij, x):
    n = len(x)
#    A = np.column_stack([i**2
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
    if len(lines) < 2:
        return np.zeros(2) * np.nan
    lines = np.array(lines)
    n = len(lines)
    pts = []
    for i in range(n):
        for j in range(i + 1, n):
            pts.append(find_intersection(lines[[i, j]]))
    out = np.mean(pts, axis=0)
    if np.isscalar(out):
        print(lines)
        print('average_intersection', out)
    return out
### create all lines:
hori = []
vert = []
left = []
righ = []
for start in range(8):
    hori.append(np.array([hori_vanish_id] + [start * 8 + i for i in range(8)]))
    vert.append(np.array([vert_vanish_id] + [start + 8 * i for i in range(8)]))

for start in np.arange(8):
    left.append(np.array([left_vanish_id] +
                         [8 * start + 9 * i for i in range(8 - start)]))
    righ.append(np.array([righ_vanish_id] +
                         [8 * start - 7 * i for i in range(start + 1)]))
for start in range(1, 8):
    left.append(np.array([left_vanish_id] +
                         [start + 9 * i for i in range(8 - start)]))
    righ.append(np.array([righ_vanish_id] +
                         [8 * start + 7 + i * 7 for i in range(8 - start)]))
for diags in [hori, vert, left, righ]:
    for line in diags:
        line.sort()

def solve_diag(img, line, ids, centers, color, width, draw=False):
    n = len(ids)
    keep = [i for i in range(n) if ids[i] in line]
    ids = ids[keep]
    centers = centers[keep]
    if len(keep) > 1:
        a, b = linfit(centers)
        x = np.array([0, IM_WIDTH])
        y = (a * x + b).astype(int)
        if draw:
            cv2.line(img, (x[0], y[0]), (x[1], y[1]), color, width)
        out = np.array([a, b])
    else:
        out = np.nan * np.zeros(2)
    return out

def solve_intersection(ab, cd):
    a, b = ab
    c, d = cd
    if np.abs(a - c) < 1e-6:
        #parallel
        out = [np.inf, np.inf]
    else:
        x = (d - b) / (a - c)
        y = a * x + b
        out = np.array([x, y])
    return out

def draw_perspective(img, ids, centers):
    ### find vanishing point for columns
    if False:
        for center in centers:
            cv2.circle(img, tuple(center.astype(int)), 4, BLUE, 4)
    todo = set(range(64))
    todo = todo - set(ids)
    if len(todo) < 1:
        return
    algs = [board_map[id][0] for id in ids]
    coords = np.array([val_to_coords(id) for id in ids])

    cols = coords[:,0]
    rows = coords[:,1]
    row_lines = []
    ## center = f(coord)
    for r in set(rows):
        keep = np.logical_and(rows == r, ids >= 0)
        line = centers[keep]
        if len(line) > 1:
            a, b = linfit(centers[keep])

            row_lines.append([a, b])
            x = np.array([0, IM_WIDTH])
            y = (a * x + b).astype(int)
            #cv2.line(img, (x[0], y[0]), (x[1], y[1]), BLUE, 1)

    vanish_rows = average_intersection(row_lines).astype(int)
    if np.any(np.isnan(vanish_rows)):
        pass
    else:
        board_map[hori_vanish_id][2] = vanish_rows
        
    #cv2.circle(img, tuple(vanish_rows), 4, (56, 123, 26), 4)
    col_lines = []
    for c in set(cols):
        keep = np.logical_and(cols == c, ids >= 0)
        line = centers[keep]
        if len(line) > 1:
            a, b = linfit(centers[keep])
            col_lines.append([a, b])
            x = np.array([0, IM_WIDTH])
            y = (a * x + b).astype(int)
            if False: # c > 4
                print('coords[keep]:\n', coords[keep])
                for c in centers[keep]:
                    cv2.circle(img, tuple(c.astype(int)), 8, CYAN, 4)
                cv2.line(img, (x[0], y[0]), (x[1], y[1]), CYAN, 1)
     
            
    #vanish_cols = average_intersection(col_lines).astype(int)
    vanish_cols = np.arange(2) * np.nan
    if np.any(np.isnan(vanish_cols)):
        pass
    else:
        board_map[vert_vanish_id][2] = vanish_cols
        
    left_ab = []
    for d in left:
        ab = solve_diag(img, d, ids, centers, GREEN, 1, draw=False)
        if np.any(np.isnan(ab)):
            pass
        else:
            left_ab.append(ab)
    vanish_left = average_intersection(left_ab).astype(int)
    if np.any(np.isnan(vanish_left)):
        pass
    else:
        board_map[left_vanish_id][2] = vanish_left
    #centers = np.array(list(centers) + [vanish_left])
    #ids = np.array(list(ids) + [left_vanish_id])
    for d in left:
        keep = [i for i in range(len(ids)) if ids[i] in d]
        if len(keep) == 1:
            i = ids[keep[0]]
            c = centers[keep[0]]
            xy = c.astype(int)
            uw = vanish_left.astype(int)
            a, b = linfit(np.vstack([xy, uw]))
            x = np.array([0, IM_WIDTH])
            y = (a * x + b).astype(int)

            # cv2.line(img, (x[0], y[0]), (x[1], y[1]), GREEN, 2)

    righ_ab = []
    for d in righ:
        ab = solve_diag(img, d, ids, centers, RED, 1, draw=False)
        if np.any(np.isnan(ab)):
            pass
        else:
            righ_ab.append(ab)
    vanish_righ = average_intersection(righ_ab).astype(int)
    if np.any(np.isnan(vanish_righ)):
        pass
    else:
        board_map[righ_vanish_id][2] = vanish_righ
    #centers = np.array(list(centers) + [vanish_righ])
    #ids = np.array(list(ids) + [righ_vanish_id])
    for d in righ:
        keep = [i for i in range(len(ids)) if ids[i] in d]
        if len(keep) == 1:
            i = ids[keep[0]]
            c = centers[keep[0]]
            xy = c.astype(int)
            uw = vanish_righ.astype(int)
            a, b = linfit(np.vstack([xy, uw]))
            x = np.array([0, IM_WIDTH])
            y = (a * x + b).astype(int)

            #cv2.line(img, (x[0], y[0]), (x[1], y[1]), RED, 2)


    righ_ab = []
    for d in righ:
        ab = solve_diag(img, d, ids, centers, (255, 12, 123), 1, False)
        righ_ab.append(ab)
    vanish_righ = average_intersection(righ_ab).astype(int)
        
    for h in hori:
        ab = solve_diag(img, h, ids, centers, (255, 12, 123), 1, False)
        if np.any(np.isnan(ab)):
            continue
        h = set(h)
        for v in vert:
            cd = solve_diag(img, v, ids, centers, (255, 12, 123), 1, False)
            if np.any(np.isnan(cd)):
                continue
            v = set(v)
            intersection = todo.intersection(h.intersection(v))
            if len(intersection) > 0:
                todo = todo - intersection
                x, y = solve_intersection(ab, cd).astype(int)
                id = intersection.pop()
                board_map[id][2] =  np.array([x, y])
                cv2.circle(img, (x, y), 10, RED, 4)

            
    for h in hori:
        ab = solve_diag(img, h, ids, centers, (255, 12, 123), 1, False)
        if np.any(np.isnan(ab)):
            continue
        h = set(h)
        for l in left:
            cd = solve_diag(img, l, ids, centers, (255, 12, 123), 1, False)
            if np.any(np.isnan(cd)):
                continue
            l = set(l)
            intersection = todo.intersection(h.intersection(l))
            if len(intersection) > 0:
                todo = todo - intersection
                x, y = solve_intersection(ab, cd).astype(int)
                id = intersection.pop()
                board_map[id][2] = np.array([x, y])
                cv2.circle(img, (x, y), 10, GREEN, 4)
        for r in righ:
            cd = solve_diag(img, r, ids, centers, (255, 12, 123), 1, False)
            if np.any(np.isnan(cd)):
                continue
            r = set(r)
            intersection = todo.intersection(h.intersection(r))
            if len(intersection) > 0:
                todo = todo - intersection
                x, y = solve_intersection(ab, cd).astype(int)
                id = intersection.pop()
                board_map[id][2] =  np.array([x, y])
                cv2.circle(img, (int(x), int(y)), 10, BLUE, 4)

    for v in vert:
        ab = solve_diag(img, v, ids, centers, (255, 12, 123), 1, False)
        if np.any(np.isnan(ab)):
            continue
        v = set(v)
        for l in left:
            cd = solve_diag(img, l, ids, centers, (255, 12, 123), 1, False)
            if np.any(np.isnan(cd)):
                continue
            l = set(l)
            intersection = todo.intersection(v.intersection(l))
            if len(intersection) > 0:
                todo = todo - intersection
                x, y = solve_intersection(ab, cd).astype(int)
                id = intersection.pop()
                board_map[id][2] = np.array([x, y])
                cv2.circle(img, (int(x), int(y)), 10, (56, 123, 26), 4)
        for r in righ:
            cd = solve_diag(img, r, ids, centers, (255, 12, 123), 1, False)
            if np.any(np.isnan(cd)):
                continue
            r = set(r)
            intersection = todo.intersection(v.intersection(r))
            if len(intersection) > 0:
                todo = todo - intersection
                x, y = solve_intersection(ab, cd).astype(int)
                id = intersection.pop()
                board_map[id][2] = np.array([x, y])
                cv2.circle(img, (int(x), int(y)), 10, (56, 123, 26), 4)
    for r in righ:
        ab = solve_diag(img, v, ids, centers, (255, 12, 123), 1, False)
        if np.any(np.isnan(ab)):
            continue
        r = set(r)
        for l in left:
            cd = solve_diag(img, l, ids, centers, (255, 12, 123), 1, False)
            if np.any(np.isnan(cd)):
                continue
            l = set(l)
            intersection = todo.intersection(r.intersection(l))
            if len(intersection) > 0:
                todo = todo - intersection
                x, y = solve_intersection(ab, cd).astype(int)
                id = intersection.pop()
                board_map[id][2] = np.array([x, y])
                cv2.circle(img, (int(x), int(y)), 10, (56, 123, 26), 4)

FLIP_THRESH = 250
def check_flip():
    balance = 0
    for letter in 'abcdefgh':
        for row in [1, 2]:
            balance += np.mean(crop_square(rect, f'{letter}{row}')[0])
            balance -= np.mean(crop_square(rect, f'{letter}{9-row}')[0])
    return balance < FLIP_THRESH

def get_side():
    balance = 0
    for j in [1, 2]:
        for i in range(1, 9):
            balance += np.mean(crop_abs_square(rect, i, j)[0])
            balance -= np.mean(crop_abs_square(rect, i, 9-j)[0])
            draw_abs_square(rect, i, j, RED, 5)
            draw_abs_square(rect, i, 9-j, BLUE, 5)
    if balance > 0:
        out = chess.BLACK
    else:
        out = chess.WHITE
    return out

def findChessboardCorners(n_ave=10):
    all_corners = []
    while len(all_corners) < n_ave:
        ret, frame = vid.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7, 7))
        if ret:
            font = cv2.FONT_HERSHEY_SIMPLEX
            corners = corners.squeeze()
            sort = corners[:,0] + IM_HEIGHT * corners[:,1]
            corners = corners[np.argsort(sort)]
            corners = corners.reshape((7, 7, 2))
            for i, row in enumerate(corners):
                row = row[np.argsort(row[:,0])]
                corners[i] = row
            all_corners.append(corners)
    all_corners = np.array(all_corners)
    corners = np.mean(all_corners, axis=0)
    return corners

def centerup():
    print("Center board in field of view.  Press 'q' to continue.")
    while 1:
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

def calibrate():
    # Capture the video frame
    centerup()
    corners = findChessboardCorners()
    ij = np.empty((49, 2))
    xy = np.empty((49, 2))
    for i, row in enumerate(corners):
        for j, pos in enumerate(row):
            pos = tuple(pos.astype(int))
            #cv2.circle(gray, pos, 10, (56, 123, 26), 4)
            #image = cv2.putText(gray, f'{i}{j}', pos, font, 
            #                    1, RED, 1, cv2.LINE_AA)
            ij[i * 7 + j] = i, j
            xy[i * 7 + j] = pos
    coeff = fit(ij, xy)
    coords = np.array([[-1, -1],
                       [-1,  7],
                       [ 7,  7],
                       [ 7, -1]])[::-1]
    coords = np.array([[-1.5, -1.5],
                       [-1.5,  7.5],
                       [ 7.5,  7.5],
                       [ 7.5, -1.5]])[::-1]
    bbox = predict(coords, coeff).astype(int)
    input_pts = np.float32(np.roll(bbox, 0, axis=0))
    output_pts = np.float32([[0, 0],
                             [0, IM_HEIGHT - 1],
                             [IM_HEIGHT - 1, IM_HEIGHT - 1],
                             [IM_HEIGHT - 1, 0]])
    print(input_pts)
    print(output_pts)
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    while True:
        ret, frame = vid.read()
        rect = cv2.warpPerspective(frame, M, (IM_HEIGHT, IM_HEIGHT),
                                   flags=cv2.INTER_LINEAR)        

        cv2.imshow('rect', rect)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
    print('Calibaration complete.  You may now set up board.')
    return M, coeff

cal_npz = 'perspective_matrix.npz'
if args.calibrate:
    print('Calibrate')
    perspective_matrix, ij_coeff = calibrate()
    np.savez(cal_npz, perspective_matrix=perspective_matrix)
    print('wrote', cal_npz)
else:
    print("Skipping calibration")

perspective_matrix = np.load(cal_npz)['perspective_matrix']



game_on = False
__last_move = False

def get_rect():
    ret, frame = vid.read()
    rect = cv2.warpPerspective(frame, perspective_matrix,
                               (IM_HEIGHT, IM_HEIGHT),
                               flags=cv2.INTER_LINEAR)
    return rect

pgr = PygameRender(size=475)

def render(renderer, board, side, colors):
    thread = threading.Thread(target=renderer.render,
                              args=(board,side==chess.BLACK and not flip_board),
                              kwargs={'colors':colors},
                              daemon=True)
    thread.start()

rect = get_rect()
dark_green = '#aaaaaa'
dark_green = '#66aa66'
dark_red = '#aa3333'
colors = {'square dark':dark_red,
          'square light':"#bbbbbb",
          'square light lastmove':'#ffaaaa',
          'square dark lastmove':"#bb6666",
          'margin':'#cccccc',
          'coord':dark_red}
          

def update_camera_view(rect):
    if flip_board:
        view = rect[::-1, ::-1]
    else:
        view = rect
    cv2.imshow('view', view)
    
side = get_side()
render(pgr, board, side==chess.BLACK, colors=colors)

mqtt_clock_reset(initial_seconds, initial_increment)

while True:
    key = chr(cv2.waitKey(1) & 0xFF)
    mqtt_events = mqtt_handle_events()
    clock_hit = False
    if 'capture_queen.turn' in mqtt_events:
        ### TODO: handle more than one event
        turn = int(mqtt_events['capture_queen.turn'][0])
        if turn == 3: ### pre-game
            key = 'r'
        clock_hit = turn < 2
    rect = get_rect()
    if not game_on:
        for i in [1, 2]:
            for letter in 'abcdefgh':
                color = WHITE
                draw_square(rect, f'{letter}{i}', WHITE, 2)
                draw_square(rect, f'{letter}{9-i}', GRAY, 2)

        bbox = get_board_bbox().astype(int)
        cv2.rectangle(rect, tuple(bbox[0]), tuple(bbox[2]), WHITE, 1)
        update_camera_view(rect)
    if key == 'q':
        break
    if key == 'x' or clock_hit:
        if not game_on:
            rect = get_rect()
            ### show the plane board
            update_camera_view(rect)
                        
        game_on = True
        print('clock')
        for i in range(10):
            ret, frame = vid.read() ## clear buffer
            
        png = f'captures/{move_number:04d}.png'
        move_number += 1
        rect = cv2.warpPerspective(frame, perspective_matrix,
                                   (IM_HEIGHT, IM_HEIGHT),
                                   flags=cv2.INTER_LINEAR)
        cv2.imwrite(png, rect)
        move = find_move(rect)
        if move:
            __last_move = move
            fen = board.fen()
            open('.fen', 'w').write(fen)
            mqtt_publish_fen(fen)
            draw_square(rect, __last_move[0:2], RED, 1)
            draw_square(rect, __last_move[2:4], RED, 1)
            update_camera_view(rect)

        # put render in background thread so that image-capture is not blocked
        render(pgr, board, side, colors)
        
    if not game_on:
        pass
        #print(['White', 'Black'][side == chess.BLACK])
                
    if not game_on and key == 's':
        if side == chess.WHITE:
            side = chess.BLACK
        else:
            side = chess.WHITE
        render(pgr, board, side, colors)
    if key == 'f':
        flip_board = not flip_board
        update_camera_view(rect)
        render(pgr, board, side, colors)
        
    if game_on and key == 'r':
        ### restart
        print('restart')
        mqtt_clock_reset(initial_seconds, initial_increment)        
        board = chess.Board()
        render(pgr, board, side, colors)        
        game_on = False
        
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
 
