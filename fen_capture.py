import copy
import sys
import argparse
import threading
import os.path
import time
import re
import numpy as np
import cv2
import glob

import chess
from board_map import fit, predict
from pygame_render import PygameRender
import pgn_upload
from mqtt_clock_client import mqtt_subscribe, mqtt_start, mqtt_clock_reset
from mqtt_clock_client import MqttRenderer, mqtt_clock_pause
from mqtt_clock_client import mqtt_setblack_ms, mqtt_setwhite_ms

class ClockMove:
    def __init__(self, move, white_ms, black_ms):
        self.move = move
        self.white_ms = white_ms
        self.black_ms = black_ms
    def __str__(self):
        return f'{self.uci()}//{self.white_ms}//{self.black_ms}'
    def uci(self):
        return self.move.uci()
    
class ClockBoard:
    def __init__(self):
        self.board = chess.Board()
        self.move_stack = []
    def __repr__(self):
        return self.fen()
    def __str__(self):
        if len(self.move_stack) > 0:
            out = f'{str(self.board)}\n{self.move_stack[-1]}'
        else:
            out = f'{str(self.board)}\n-/{initial_seconds * 1000}/{initial_seconds*1000}'
        return out
    def __len__(self):
        return len(self.move_stack)
    def fen(self):
        return self.board.fen()
    
    def push(self, clock_move):
        self.move_stack.append(clock_move)
        return self.board.push(clock_move.move)

    def push_uci(self, uci, white_ms, black_ms):
        move = chess.Move.from_uci(uci)
        clock_move = ClockMove(move, white_ms, black_ms)
        self.push(clock_move)

    def pop(self):
        self.board.pop() ### keep in sync
        return self.move_stack.pop()
    
    @property
    def legal_moves(self):
        return self.board.legal_moves
    
    def is_castling(self, move):
        return self.board.is_castling(move)
    
    def is_en_passant(self, move):
        return self.board.is_en_passant(move)

    def copy(self):
        out = ClockBoard()
        out.board = self.board.copy()
        out.move_stack = copy.copy(self.move_stack)
        return out

for file in glob.glob('captures/*.png'):
    os.remove(file)
game_id = 0

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
    'u' to upload to lichess
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

clock_board = ClockBoard()

open('.fen', 'w').write(clock_board.fen())

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
    for move in clock_board.legal_moves:
        ## only allow queen promotion at this time
        if move.promotion and move.promotion != chess.QUEEN:
            continue
        uci = move.uci()
        sq0, bbox0 = crop_square(thresh, uci[:2])
        sq1, bbox1 = crop_square(thresh, uci[2:4])
        sqs = [sq0, sq1]
        if clock_board.is_castling(move):
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
        if clock_board.is_en_passant(move):
            col = uci[2]
            row = uci[1]
            sq2, bbox2 = crop_square(thresh, f'{col}{row}')
            sqs.append(sq2)
            
        change = np.array([int(np.sum(sq)) for sq in sqs])
        total_change = np.sum(change)
        change_thresh = 5000
        change_count = np.sum(change > change_thresh)
        #print(uci, move.from_square, move.to_square, move.promotion,
        #      clock_board.is_en_passant(move),
        #      clock_board.is_castling(move), change_count)
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
        
        clock_board.push_uci(out, white_ms, black_ms)
        print(clock_board.fen())
        print(clock_board.fen(), file=open(".fen", 'w'), flush=True)
        
        
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

move_number = 0

DEG = np.pi / 180
grid_guess = [-27214, -3834, 43305, 2.2340, 0.8692, -1.0829]

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

def findChessboardCorners(n_ave=10, max_tries=20):
    print('Locating chessboard...')
    all_corners = []
    iter = 0
    while len(all_corners) < n_ave and iter < max_tries:
        print(f'iter: {iter}/{max_tries} {len(all_corners)}/{n_ave}')
        ret, frame = vid.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7, 7))
        #ret, corners = cv2.findChessboardCorners(gray, (3, 7))
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
        iter += 1
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

def render(renderer, board, side, colors):
    thread = threading.Thread(target=renderer.render,
                              args=(board,side==chess.BLACK and not flip_board),
                              kwargs={'colors':colors},
                              daemon=True)
    thread.start()

class Renderers:
    def __init__(self, render_list):
        self.renderers = render_list
    def render(self, board, side, colors):
        for renderer in self.renderers:
            renderer.render(board, side, colors=colors)

mqttr = MqttRenderer
#pgr = PygameRender(size=475)
#renderers = Renderers([pgr, mqttr])
renderers = Renderers([mqttr])

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
render(renderers, None, side==chess.BLACK, colors=colors)

mqtt_clock_reset(initial_seconds, initial_increment)

old_board = None

def get_position_image_filename(game_id, move_numner):
    return f'captures/{game_id:04d}_{move_number:04d}.png'
def write_position_image(rect, game_id, move_number):
    print('write', game_id, move_number)
    png = get_position_image_filename(game_id, move_number)
    cv2.imwrite(png, rect)    
def read_position_image(game_id, move_number):
    png = get_position_image_filename(game_id, move_number)
    if os.path.exists(png):
        print('read', game_id, move_number, png)
        out = cv2.imread(png)
    else:
        print('XXXX', game_id, move_number, png)
        out = None
    return out


next_white_ms = initial_seconds * 1000
next_black_ms = initial_seconds * 1000
while True:
    key = chr(cv2.waitKey(1) & 0xFF)
    mqtt_events = mqtt_handle_events()
    clock_hit = False
    # print(key)
    if key == 'Q':
        game_on = False
        mqtt_clock_pause(True)
        if len(clock_board.move_stack) > 0:
            if old_board is None:
                old_board = clock_board.copy()
                print('going back...total moves available:',
                      len(old_board.move_stack))
            clock_move = clock_board.pop()
            mqtt_setwhite_ms(clock_move.white_ms)
            mqtt_setblack_ms(clock_move.black_ms)
        else:
            clock_move = None
        render(renderers, clock_board, side, colors)
        move_number = len(clock_board.move_stack)
        print('go back', move_number)

        im = read_position_image(game_id, len(clock_board.move_stack))
        if im is not None:
            if clock_move is not None:
                uci = clock_move.uci()
                draw_square(im, uci[0:2], RED, 1)
                draw_square(im, uci[2:4], RED, 1)
            
            cv2.imshow("Previous Moves", im)
    if key == 'S':
        if not game_on and old_board is not None:
            n = len(clock_board.move_stack)
            m = len(old_board.move_stack)
            print(m, n)
            if  m > n:
                clock_move = old_board.move_stack[n]
                if len(old_board) > n + 1:
                    next_move = old_board.move_stack[n+1]
                else:
                    next_move = ClockMove('xxxx', next_white_ms, next_black_ms)
                clock_board.push(clock_move)
                mqtt_setwhite_ms(next_move.white_ms)
                mqtt_setblack_ms(next_move.black_ms)
                move_number = n + 1
                im = read_position_image(game_id, move_number)
                if im is not None:
                    uci = clock_move.uci()
                    draw_square(im, uci[0:2], RED, 1)
                    draw_square(im, uci[2:4], RED, 1)
                    cv2.imshow("Previous Moves", im)
                
            render(renderers, clock_board, side, colors)
    if key == 'u':
        pgn_upload.upload_to_lichess(clock_board)
    if 'capture_queen.turn' in mqtt_events:
        ### TODO: handle more than one event
        turn_msg = str(mqtt_events['capture_queen.turn'][0])[2:-1]
        white_ms = next_white_ms
        black_ms = next_black_ms
        turn, next_white_ms, next_black_ms = map(int, turn_msg.split('//'))
        clock_hit = True
        old_board = None
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
            last_rect = rect.copy()
            update_camera_view(rect)
        game_on = True
        cv2.destroyWindow("Previous Moves")
        for i in range(1):
            ret, frame = vid.read() ## clear buffer
            
        png = f'captures/{game_id:04d}_{move_number:04d}.png'
        rect = cv2.warpPerspective(frame, perspective_matrix,
                                   (IM_HEIGHT, IM_HEIGHT),
                                   flags=cv2.INTER_LINEAR)
        move = find_move(rect)
        move_number = len(clock_board.move_stack)
        write_position_image(rect, game_id, move_number)

        if move:
            __last_move = move
            fen = clock_board.fen()
            open('.fen', 'w').write(fen)
            draw_square(rect, __last_move[0:2], RED, 1)
            draw_square(rect, __last_move[2:4], RED, 1)
            update_camera_view(rect)

        # put render in background thread so that image-capture is not blocked
        render(renderers, clock_board, side, colors)
        
    if not game_on:
        pass
        #print(['White', 'Black'][side == chess.BLACK])
                
    if not game_on and key == 's':
        if side == chess.WHITE:
            side = chess.BLACK
        else:
            side = chess.WHITE
        render(renderers, clock_board, side, colors)
    if key == 'f':
        flip_board = not flip_board
        update_camera_view(rect)
        render(renderers, clock_board, side, colors)
    if 'capture_queen.reset_pi' in mqtt_events:
        next_white_ms = initial_seconds * 1000
        next_black_ms = initial_seconds * 1000
        old_board = None
        print('restart')
        clock_board = ClockBoard()
        render(renderers, None, side, colors)        
        game_on = False
        
    if key == 'r':
        ### restart
        next_white_ms = initial_seconds * 1000
        next_black_ms = initial_seconds * 1000
        old_board = None
        print('restart')
        game_id += 1
        clock_board = ClockBoard()
        mqtt_clock_reset(initial_seconds, initial_increment)
        render(renderers, None, side, colors)        
        game_on = False
        
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
 
