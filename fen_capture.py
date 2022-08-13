import glob
import time
import re
import numpy as np
import cv2
import cv2.aruco as aruco
import chess
from grid_perspective import fit, ChessBoard, Camera
from board_map import fit, predict

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

def crop_square(frame, alg, coeff, draw=False):
    i, j = val_to_coords(alg_map[alg])
    coords = np.array([[-1, -1],
                       [-1,  1],
                       [ 1,  1],
                       [ 1, -1]]) / 2 + np.array([i, j])
    bbox = predict(coords, coeff).astype(int)
    starts = np.min(bbox, axis=0).astype(int)
    stops = np.max(bbox, axis=0).astype(int) +1
    bbox = bbox.reshape((1, -1, 1, 2)).astype(np.int32)
    if draw:
        cv2.polylines(frame, bbox, True, (0, 0, 255), 1)
    return frame[starts[1]:stops[1],starts[0]:stops[0]], bbox
        
def val_to_coords(val):
    row, col = divmod(val, 8)
    return (col, 7 - row)

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
PURPLE = (255, 0, 355)

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

last_frame = None
def find_move(frame, free):
    global last_frame
    if last_frame is not None:
        delta = cv2.absdiff(frame, last_frame)
        imgray = cv2.cvtColor(delta,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,50,255,0)
        cv2.imshow('squares', np.transpose(frame, (1, 0, 2))[:,::-1])
    else:
        thresh = None
    last_frame = frame.copy()
    xy = np.array([board_map[id][2] for id in board_map if id >= 0 and
                   board_map[id][2] is not None])
    ij = np.array([val_to_coords(id) for id in board_map if id >= 0 and
                   board_map[id][2] is not None])
    if len(np.shape(xy)) < 2:
        return
    if len(xy) < 1:
        return
    keep = np.logical_not(np.isnan(xy[:,0]))
    xy = xy[keep]
    ij = ij[keep]
    coeff = fit(ij, xy)

    start = time.time()
    changes = []
    if thresh is None:
        return

    candidates = []
    for move in board.legal_moves:
        ## only allow queen promotion at this time
        if move.promotion and move.promotion != chess.QUEEN:
            continue
        uci = move.uci()
        sq0, bbox0 = crop_square(thresh, uci[:2], coeff)
        sq1, bbox1 = crop_square(thresh, uci[2:4], coeff)
        sqs = [sq0, sq1]
        if board.is_castling(move):
            if uci[2] == 'g': ## kingside
                row = uci[3]
                sq2, bbox2 = crop_square(thresh, f'h{row}', coeff)
                sq3, bbox3 = crop_square(thresh, f'f{row}', coeff)
                sqs.extend([sq2, sq3])
            if uci[2] == 'c': ## queen
                row = uci[3]
                sq2, bbox2 = crop_square(thresh, f'a{row}', coeff)
                sq3, bbox3 = crop_square(thresh, f'd{row}', coeff)
                sqs.extend([sq2, sq3])
        if board.is_en_passant(move):
            col = uci[2]
            row = uci[1]
            sq2, bbox2 = crop_square(thresh, f'{col}{row}', coeff)
            sqs.append(sq2)
            
        change = np.array([int(np.sum(sq)) for sq in sqs])
        total_change = np.sum(change)
        change_thresh = 25000
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
        sq, bbox = crop_square(frame, out[:2],
                               coeff, draw=True)
        sq, bbox = crop_square(frame, out[2:4],
                               coeff, draw=True)
        #cv2.imshow('piece', sq)
        cv2.imshow('squares', np.transpose(frame, (1, 0, 2))[:,::-1])
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

            

npzs = glob.glob('CameraCalibrate/results/*.npz')
npzs.sort()
npz = npzs[-1]
results = np.load(npz)
camera_matrix = results['camera_matrix']
distorition_coeff = results['distorition_coeff']
side = 2000
virtual_board = ChessBoard(side, 'g', 'w')

while(True):
    # Capture the video frame
    ret, frame = vid.read()

    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #frame = cv2.undistort(frame,camera_matrix,distorition_coeff,None)
    # cv2.imwrite("img/board.png", frame)
    bboxs, free_ids = findArucoMarkers(frame, draw=False)
    if len(free_ids) == 0:
        continue
    centers = np.mean(bboxs, axis=-2).squeeze()
    focal_length = 1000
    side = 1000
    centers = []
    ids = []
    for id in board_map:
        bbox = board_map[id][1]
        center = board_map[id][2]
        if bbox is not None:
            center = np.mean(bbox, axis=0)
            centers.append(center)
            ids.append(id)
        elif center is not None:
            ids.append(id)
            centers.append(center)
    ids = np.array(ids)
    centers = np.array(centers)
    if len(free_ids) > 0:
        guess = draw_perspective(frame, ids, centers)
    cv2.imshow('frame', frame)# [::-1,::-1])
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('x'):
        print('clock')
        for i in range(10):
            ret, frame = vid.read() ## clear buffer
            
        png = f'captures/{move_number:04d}.png'
        move_number += 1
        cv2.imwrite(png, frame)
        move = find_move(frame, free_ids)
        if move:
            open('.fen', 'w').write(board.fen())
    if key & 0xFF == ord('w'):
        print('code here for special handling')
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
 
