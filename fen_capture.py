from collections import defaultdict
import json
from datetime import datetime
from datetime import date
import requests
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


from mqtt_clock_client import mqtt_subscribe, mqtt_start, mqtt_clock_reset
from mqtt_clock_client import MqttRenderer, mqtt_clock_pause
from mqtt_clock_client import mqtt_setblack_ms, mqtt_setwhite_ms
from mqtt_clock_client import mqtt_sethalfmove, mqtt_game_over

import pgn_upload
import chess_db
from board_map import fit, predict
import kicker
import defaults

desc = 'Capture Queen: Over-the-board real-time chess capture system.'
shortcuts = '''\
During game play, these keys are active:
    's' to swap colors
    'f' to flip sides
    'r' to reset to new game
    'q' to quit
    'x' to make move
    'u' to upload to lichess
  right to go back a move
   left to go forward a move
'''

parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-c','--calibrate',
                    help='Calibrate board area',
                    required=False, default=False)
parser.add_argument('-d','--display',
                    help='Display board area',
                    required=False, default="True")
parser.add_argument('-s','--shortcuts',
                    help='get gameplay command keys',
                    action="store_true")
parser.add_argument('-w','--white',
                    help='white player name (lichess id)',
                    required=False, default="{white_player}")
parser.add_argument('-b','--black',
                    help='black player name (lichess id)',
                    required=False, default="{black_player}")
parser.add_argument('-t','--time_control',
                    help='Time control seconds+increment',
                    required=False, default=defaults.time_control)
args = parser.parse_args()
if args.shortcuts:
    print(shortcuts)
    sys.exit()

display_on = args.display == "True"

white_player = args.white
black_player = args.black
time_control = args.time_control
initial_seconds, increment_seconds = map(int, time_control.split('+'))

def imshow(name, frame):
    if display_on:
        cv2.imshow(name, frame)

### register with wyolum
def getip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]

localip = getip()
utc_url = ('https://wyolum.com/utc_offset/utc_offset.py'
           '?dev_type=CaptureQueen.Mosquitto&'
           f'localip={localip}')
r = requests.get(utc_url, headers={"User-Agent":"XY"})
utc_offset_response = json.loads(r.content.decode('utf-8'))

print('Registered mqtt server with wyolum.com')
   
class ClockMove:
    def __init__(self, move, white_ms, black_ms):
        self.move = move
        self.white_ms = white_ms
        self.black_ms = black_ms
    def __getattr__(self, name):
        return getattr(self.move, name)
    def __str__(self):
        return f'{self.uci()}//{self.white_ms}//{self.black_ms}'
    def uci(self):
        return self.move.uci()

class ClockBoard:
    def __init__(self, initial_seconds, increment_seconds):
        self.initial_seconds = initial_seconds
        self.increment_seconds = increment_seconds
        self.__resign = None
        
        self.board = chess.Board()
        self.move_stack = []
        self.headers = defaultdict(lambda :"?")
        self.start_time = datetime.now()
        datestr = self.start_time.strftime("%d/%m/%y %H:%M:%S")
        self.gameid = chess_db.get_gameid(datestr)
        self.headers['Date'] = datestr

        region = utc_offset_response['region']
        city = utc_offset_response['city']
        time_control = f'{initial_seconds}+{increment_seconds}'
        self.headers['Event'] =  'CaptureQueen Over-the-board chess capture'
        self.headers['Site'] = f'{city},{region}'
        self.headers['White'] = f'{white_player}'
        self.headers['Black'] = f'{black_player}'
        self.headers['Annotator'] = 'CaptureQueen'
        self.headers['TimeControl'] = f'{time_control}'
        self.headers['Variant'] = 'Standard'
        chess_db.update_game(self.gameid, self.headers)

    def resign(self, color):
        self.__resign = color
        if color == chess.WHITE:
            color = 'White'
            if len(self.move_stack) > 0:
                ms = self.move_stack[-1].white_ms
            else:
                ms = self.initial_seconds * 1000
        elif color == chess.BLACK:
            color = 'Black'
            if len(self.move_stack) > 0:
                ms = self.move_stack[-1].black_ms
            else:
                ms = self.initial_seconds * 1000
        else:
            raise ValueError(f'Unknown color "{color}"')
        result = self.get_result()
        chess_db.move(self.gameid, self.ply(), f'{{{color} resigns.}} {{{ms/1000:<7.1f}}} {result}')
        mqtt_game_over(result)
        pgn_upload.upload_to_lichess(self.get_pgn())

        
    def get_result(self):
        out = ''
        outcome = self.outcome()
        if self.__resign is not None:
            if self.__resign == chess.WHITE:
                out = '0-1'
            elif self.__resign == chess.BLACK:
                out = '1-0'
        elif outcome:
            winner = outcome.winner
            if winner is not None:
                if winner == chess.WHITE:
                    out = "1-0"
                else:
                    out = "0-1"
            if len(self.move_stack) > 0:
                last_move = self.move_stack[-1]
                white_ms = last_move.white_ms
                black_ms = last_move.black_ms
                if white_ms <= 0 and black_ms > 0:
                    out = "0-1"
                if black_ms <= 0 and white_ms > 0:
                    out = "1-0"
        return out

    def get_termination(self):
        outcome = self.outcome()
        if self.__resign is not None:
            out = 'Normal'
        elif outcome:
            out = str(outcome.termination).split('.')[1]
            if len(self.move_stack) > 0:
                last_move = self.move_stack[-1]
                white_ms = last_move.white_ms
                black_ms = last_move.black_ms
                if white_ms <= 0 and black_ms > 0:
                    out = "Time forfeit"
                if black_ms <= 0 and white_ms > 0:
                    out = "Time forfeit"
        else:
            out = 'None'
        return out

    def get_pgn(self):
        board = chess.Board()
        moves = self.move_stack
        self.headers['Result'] = self.get_result()
        self.headers['Termination'] = self.get_termination()
        chess_db.update_game(self.gameid, self.headers)
        return chess_db.get_pgn(self.gameid)
    
    def set_timeout(self, white_ms, black_ms):
        legal_moves = list(self.legal_moves)
        if len(legal_moves) > 0:
            ### make dummy move to complete game with time forfiet
            move = legal_moves[0]
            self.move_stack.append(ClockMove(move, white_ms, black_ms))

    def __getattr__(self, name):
        return getattr(self.board, name)
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
        san = self.san(clock_move)
        self.move_stack.append(clock_move)
        ply = self.ply()
        if ply % 2 == chess.WHITE:
            ms = clock_move.white_ms
        else:
            ms = clock_move.black_ms
        chess_db.move(self.gameid, ply, f'{san:>7s} {{{ms/1000:<7.1f}}}')
        return self.board.push(clock_move.move)

    def push_uci(self, uci, white_ms, black_ms):
        move = chess.Move.from_uci(uci)
        clock_move = ClockMove(move, white_ms, black_ms)
        self.push(clock_move)

    def pop(self):
        self.board.pop() ### keep in sync
        return self.move_stack.pop()
    
    def copy(self):
        out = ClockBoard(self.initial_seconds, self.increment_seconds)
        out.board = self.board.copy()
        out.move_stack = copy.copy(self.move_stack)
        out.headers = self.headers
        return out

for file in glob.glob('captures/*.png'):
    os.remove(file)
game_id = 0

mqtt_pending_msgs = []
def on_mqtt(msg):
    mqtt_pending_msgs.append(msg)

def mqtt_gather_events():
    out = {}
    while mqtt_pending_msgs:
        msg = mqtt_pending_msgs.pop()
        if msg.topic not in out:
            out[msg.topic] = []
        payload = msg.payload.decode('utf-8')
        out[msg.topic].append(payload)
        #print(msg.topic, payload)
    return out

mqtt_subscribe(on_mqtt)
mqtt_start()

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

def draw_abs_square(rectified, i, j, color, thickness):
    bbox = get_abs_bbox(i, j)
    cv2.rectangle(rectified, tuple(bbox[0]), tuple(bbox[2]), color, thickness)
    
def draw_square(rectified, alg, color, thickness):
    bbox = get_bbox(alg)
    cv2.rectangle(rectified, tuple(bbox[0]), tuple(bbox[2]), color, thickness)

def crop_abs_square(rectified, i, j):
    bbox = get_abs_bbox(i, j)
    starts = np.min(bbox, axis=0).astype(int)
    stops = np.max(bbox, axis=0).astype(int) + 1
    bbox = bbox.reshape((1, -1, 1, 2)).astype(np.int32)
    out = rectified[starts[1]:stops[1],starts[0]:stops[0]], bbox
    return out
    
def crop_square(rectified, alg):
    bbox = get_bbox(alg)
    starts = np.min(bbox, axis=0).astype(int)
    stops = np.max(bbox, axis=0).astype(int) + 1
    bbox = bbox.reshape((1, -1, 1, 2)).astype(np.int32)
    out = rectified[starts[1]:stops[1],starts[0]:stops[0]], bbox
    return out
        
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
PURPLE = (255, 0, 355)
WHITE = (255, 255, 255)
BLACK = (0, 0 ,0)
GRAY = (128, 128, 128)

clock_board = ClockBoard(initial_seconds, increment_seconds)

open('.fen', 'w').write(clock_board.fen())

last_rectified = None
def find_move(rectified):
    global last_rectified
    if last_rectified is not None:
        delta = cv2.absdiff(rectified, last_rectified)
        sum_delta = np.sum(delta)
        BUMP_THRESH = 5000000
        if sum_delta > BUMP_THRESH:
            print(f'Board moved {sum_delta} > {BUMP_THRESH}')
        thresh = np.max(delta, axis=-1)
    else:
        thresh = None
    last_rectified = rectified.copy()

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
        if False: ## cool image of changes since last image
            draw_square(thresh, out[0:2], WHITE, 1)
            draw_square(thresh, out[2:4], WHITE, 1)
            imshow("thresh", thresh) 
        clock_board.push_uci(out, white_ms, black_ms)
        fen = clock_board.fen()
        print(fen, file=open(".fen", 'w'), flush=True)
    return out

vid = cv2.VideoCapture(0)
IM_WIDTH = 640
IM_HEIGHT = 480
vid.set(cv2.CAP_PROP_FRAME_WIDTH, IM_WIDTH)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, IM_HEIGHT)

move_number = 0

FLIP_THRESH = 250
def check_flip():
    balance = 0
    for letter in 'abcdefgh':
        for row in [1, 2]:
            balance += np.mean(crop_square(rectified, f'{letter}{row}')[0])
            balance -= np.mean(crop_square(rectified, f'{letter}{9-row}')[0])
    return balance < FLIP_THRESH

def get_side():
    balance = 0
    for j in [1, 2]:
        for i in range(1, 9):
            balance += np.mean(crop_abs_square(rectified, i, j)[0])
            balance -= np.mean(crop_abs_square(rectified, i, 9-j)[0])
            draw_abs_square(rectified, i, j, RED, 5)
            draw_abs_square(rectified, i, 9-j, BLUE, 5)
    if balance > 0:
        out = chess.BLACK
    else:
        out = chess.WHITE
    return out

def findChessboardCorners(n_ave=1, max_tries=20):
    print('Locating chessboard...')
    all_corners = []
    iter = 0
    while len(all_corners) < n_ave and iter < max_tries:
        print(f'iter: {iter}/{max_tries} {len(all_corners)}/{n_ave}')
        ret, frame = vid.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7, 7))
        if ret:
            font = defaults.font
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

    ret, image = vid.read()
    for i, row in enumerate(corners):
        for j, c in enumerate(row):
            pos = tuple(c.astype(int))
            cv2.circle(image,  pos, 10, (56, 123, 26), 4)
            image = cv2.putText(image, f'{i}{j}', pos, defaults.font, 
                                1, RED, 1, cv2.LINE_AA)

    imshow('Corners', image)
    return corners

def centerup():
    print("Center board in field of view.  Press 'q' to continue.")
    font = defaults.font
    while 1:
        ret, frame = vid.read()
        frame = cv2.putText(frame, f'Calibrating camera ...',
                            (150,40), font, 
                            1, RED, 1, cv2.LINE_AA)
        frame = cv2.putText(frame, f'... clear board,',
                            (200,IM_HEIGHT//2 - 40), font, 
                            1, RED, 1, cv2.LINE_AA)
        frame = cv2.putText(frame, f'press "q" when centered.',
                            (100,IM_HEIGHT//2), font, 
                            1, RED, 1, cv2.LINE_AA)
        
        imshow('Calibrate', frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

def calibrate():
    # Capture the video frame
    centerup()
    corners = findChessboardCorners(1)

    ### extend 7x7 corners found in calibration to edge of board
    ### using a polynomial fit
    
    ij = np.empty((49, 2))
    xy = np.empty((49, 2))
    for i, row in enumerate(corners):
        for j, pos in enumerate(row):
            pos = tuple(pos.astype(int))
            ij[i * 7 + j] = i, j
            xy[i * 7 + j] = pos
    coeff = fit(ij, xy)

    # find baord edge
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
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    font = defaults.font
    while True:
        ret, frame = vid.read()
        rectified = cv2.warpPerspective(frame, M, (IM_HEIGHT, IM_HEIGHT),
                                   flags=cv2.INTER_LINEAR)        

        rectified = cv2.putText(rectified, f'Press "q" to continue.',
                                (10,IM_HEIGHT//2), font, 
                                1, RED, 1, cv2.LINE_AA)
        rectified = cv2.putText(rectified, f'Press "x" to redo.',
                                (10,IM_HEIGHT//2 + 40), font, 
                                1, RED, 1, cv2.LINE_AA)
        imshow('Calibrate', rectified)
        key = chr(cv2.waitKey(1) & 0xFF)
        if key == 'q':
            break
        if key == 'x':
            calibrate()
            break
    return M, coeff

cal_npz = 'perspective_matrix.npz'
if args.calibrate:
    print('Calibrate')
    perspective_matrix, ij_coeff = calibrate()
    np.savez(cal_npz, perspective_matrix=perspective_matrix)
    print('wrote', cal_npz)
    print('Calibaration complete.  You may now set up board.')
    cv2.destroyAllWindows()
else:
    print("Skipping calibration")

kicker.kick('pyqt_mqtt_chess.py')
    
perspective_matrix = np.load(cal_npz)['perspective_matrix']

game_on = False
__last_move = False

def get_rectified():
    ret, frame = vid.read()
    rectified = cv2.warpPerspective(frame, perspective_matrix,
                               (IM_HEIGHT, IM_HEIGHT),
                               flags=cv2.INTER_LINEAR)
    return rectified

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
renderers = Renderers([mqttr])

rectified = get_rectified()
dark_green = '#66aa66'
dark_red = '#aa3333'
colors = {'square dark':dark_red,
          'square light':"#bbbbbb",
          'square light lastmove':'#ffaaaa',
          'square dark lastmove':"#bb6666",
          'margin':'#cccccc',
          'coord':dark_red}
colors = defaults.colors          

def update_camera_view(rectified):
    if flip_board:
        view = rectified[::-1, ::-1]
    else:
        view = rectified
    imshow('view', view)
    
side = get_side()
render(renderers, None, side==chess.BLACK, colors=colors)

mqtt_clock_reset(initial_seconds, increment_seconds)

old_board = None

def get_position_image_filename(game_id, move_numner):
    return f'captures/{game_id:04d}_{move_number:04d}.png'
def write_position_image(rectified, game_id, move_number):
    png = get_position_image_filename(game_id, move_number)
    cv2.imwrite(png, rectified)    
def read_position_image(game_id, move_number):
    png = get_position_image_filename(game_id, move_number)
    if os.path.exists(png):
        out = cv2.imread(png)
    else:
        out = None
    return out


white_ms = initial_seconds * 1000
black_ms = initial_seconds * 1000
while True:
    key = chr(cv2.waitKey(1) & 0xFF)
    mqtt_events = mqtt_gather_events()
    clock_hit = False
    # print(key)
    if key == 'Q' or 'capture_queen.goback' in mqtt_events: ### go back
        game_on = False
        mqtt_clock_pause(True)
        if len(clock_board.move_stack) > 0:
            if old_board is None:
                old_board = clock_board.copy()
                print('going back...total moves available:',
                      len(old_board.move_stack))
            clock_move = clock_board.pop()
            if len(clock_board.move_stack) > 0:
                white_ms = clock_board.move_stack[-1].white_ms
                black_ms = clock_board.move_stack[-1].black_ms
            else:
                white_ms = initial_seconds * 1000
                black_ms = initial_seconds * 1000
            mqtt_setwhite_ms(white_ms)
            mqtt_setblack_ms(black_ms)
            mqtt_sethalfmove(len(clock_board.move_stack))
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
            
            #imshow("Previous Moves", im)
    if key == 'S' or 'capture_queen.goforward' in mqtt_events:
        if not game_on and old_board is not None:
            n = len(clock_board.move_stack)
            m = len(old_board.move_stack)
            if  m > n:
                clock_move = old_board.move_stack[n]
                if len(old_board) > n:
                    next_move = old_board.move_stack[n]
                else:
                    next_move = ClockMove('xxxx', white_ms, black_ms)
                clock_board.push(clock_move)
                mqtt_setwhite_ms(next_move.white_ms)
                mqtt_setblack_ms(next_move.black_ms)
                mqtt_sethalfmove(len(clock_board.move_stack))
                move_number = n + 1
                im = read_position_image(game_id, move_number)
                if im is not None:
                    uci = clock_move.uci()
                    draw_square(im, uci[0:2], RED, 1)
                    draw_square(im, uci[2:4], RED, 1)
                    #imshow("Previous Moves", im)
                
            render(renderers, clock_board, side, colors)
    if key == 'u':
        pgn_upload.upload_to_lichess(clock_board.get_pgn())
    if 'capture_queen.turn' in mqtt_events:
        ### TODO: handle more than one event
        turn_msg = mqtt_events['capture_queen.turn'][0]
        turn, white_ms, black_ms = map(int, turn_msg.split('//'))
        if black_ms <= 0 or white_ms <= 0:
            clock_board.set_timeout(white_ms, black_ms)
        clock_hit = True
        old_board = None
    
    rectified = get_rectified()
    if not game_on:
        ranks = str(clock_board).splitlines()[:8]
        for j, rank in enumerate(ranks):
            rank = rank[0::2]
            number = 8 - j
            for i, c in enumerate(rank[:8]):
                letter = 'abcdefgh'[i]
                if c != '.':
                    if c.upper() == c:
                        color = WHITE
                    else:
                        color = GRAY
                    draw_square(rectified, f'{letter}{number}', color, 2)
        #for i in [1, 2]:
        #    for letter in 'abcdefgh':
        #        draw_square(rectified, f'{letter}{i}', WHITE, 2)
        #        draw_square(rectified, f'{letter}{9-i}', GRAY, 2)

        bbox = get_board_bbox().astype(int)
        cv2.rectangle(rectified, tuple(bbox[0]), tuple(bbox[2]), WHITE, 1)
        update_camera_view(rectified)
    if 'capture_queen.resign' in mqtt_events:
        color = mqtt_events['capture_queen.resign'][0] == 'True'
        clock_board.resign(color)
        mqtt_clock_pause(True)
        
    if key == 'q' or 'capture_queen.quit' in mqtt_events:
        break
    if key == 'x' or clock_hit:
        if not game_on:
            rectified = get_rectified()
            ### show the plane board
            last_rectified = rectified.copy()
            update_camera_view(rectified)
        game_on = True
        cv2.destroyWindow("Previous Moves")
        for i in range(1):
            ret, frame = vid.read() ## clear buffer
            
        png = f'captures/{game_id:04d}_{move_number:04d}.png'
        rectified = cv2.warpPerspective(frame, perspective_matrix,
                                   (IM_HEIGHT, IM_HEIGHT),
                                   flags=cv2.INTER_LINEAR)
        move = find_move(rectified)
        move_number = len(clock_board.move_stack)
        write_position_image(rectified, game_id, move_number)

        if move:
            __last_move = move
            fen = clock_board.fen()
            open('.fen', 'w').write(fen)
            draw_square(rectified, __last_move[0:2], RED, 1)
            draw_square(rectified, __last_move[2:4], RED, 1)
            update_camera_view(rectified)

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
        update_camera_view(rectified)
        render(renderers, clock_board, side, colors)
    if 'capture_queen.reset_pi' in mqtt_events:
        white_ms = initial_seconds * 1000
        black_ms = initial_seconds * 1000
        old_board = None
        print('restart')
        fen = clock_board.fen()
        open('.fen', 'w').write(fen)
        clock_board = ClockBoard(initial_seconds, increment_seconds)
        render(renderers, None, side, colors)        
        game_on = False
        
    if key == 'r':
        ### restart
        white_ms = initial_seconds * 1000
        black_ms = initial_seconds * 1000
        old_board = None
        print('restart')
        fen = clock_board.fen()
        open('.fen', 'w').write(fen)
        game_id += 1
        clock_board = ClockBoard(initial_seconds, increment_seconds)
        mqtt_clock_reset(initial_seconds, increment_seconds)
        render(renderers, None, side, colors)        
        game_on = False
        
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
 
