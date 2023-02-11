'''
Add stereo capibility
'''
import sys
import os
import sys
import time
import pylab as pl
import numpy as np
import chess
import cv2
import argparse

from colors import RED, GREEN, BLUE, CYAN, PURPLE, WHITE, BLACK, GRAY
import defaults
import Util
from Util import linfit, intersect
import board_map
IM_WIDTH = 640
IM_HEIGHT = 480
IM_WIDTH = 800
IM_HEIGHT = 600

IM_WIDTH = 2560 ### Stereo
IM_HEIGHT = 960 ### Stereo
IM_WIDTH = 640 ### Single image
IM_HEIGHT = 480 ### 

DELTA = IM_HEIGHT / 9
BBOX = (np.array([
    [-0.5,  7.5],
    [-0.5, -0.5],
    [ 7.5, -0.5],
    [ 7.5,  7.5],
]) + 1) * DELTA



def clearCapture(capture):
    capture.release()
    cv2.destroyAllWindows()

def countCameras():
    n = 0
    for i in range(10):
        try:
            cap = cv2.VideoCapture(i)
            ret, frame = cap.read()
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clearCapture(cap)
            n += 1
        except:
            clearCapture(cap)
            break
    return n

#n_cam = countCameras() ### this hangs up the camera :-(
n_cam = 1

class ChessCam:
    ### stereo chess cam
    ### flip_board
    ### If True, the right side of board from camera view point is toward the
    ### bottom of the board.
    ### If False, the left side of the board is toward the bottom

    def __init__(self, flip_board=False, side=chess.WHITE,
                 cal_npz='perspective_matrices.npz', camera_number=0):
        self.flip_board = flip_board
        self.side = side
        if os.path.exists(cal_npz):
            self.perspective_matrices = np.load(cal_npz)['perspective_matrix']
        else:
            self.perspective_matrices = np.hstack([np.eye(4)[:3], np.eye(4)[:3]])
        self.vid = cv2.VideoCapture(camera_number)
        self.vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 2 * IM_WIDTH)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, IM_HEIGHT)
        #self.vid.set(cv2.CAP_PROP_EXPOSURE, 15)
        self.cal_npz = cal_npz

    def get_board_bbox(self):
        '''
        Get bounding box for entire board
        '''
        return BBOX

    def get_abs_bbox(self, i, j):
        '''
        Return bounding box for given square regardless of board orientation.
        '''
        coords = np.array([[-1, -1],
                           [-1,  1],
                           [ 1,  1],
                           [ 1, -1]]) / 3 + np.array([i, j]) ### to make bigger use  / 3, (/2 = full sq)
        bbox = coords * DELTA
        return bbox.astype(int)

    def alg2ij(self, alg):
        i = ord(alg[0]) - ord('a') + 1
        j = 9 - int(alg[1])
        if self.side == chess.BLACK:
            j = 9 - j
            i = 9 - i
        return i, j
    
    def get_bbox(self, alg):
        i, j = self.alg2ij(alg)
        bbox = self.get_abs_bbox(i, j)
        return bbox

    def draw_abs_square(self, left_right, i, j, color, thickness):
        bbox = self.get_abs_bbox(i, j)
        for rectified in left_right:
            cv2.rectangle(rectified, tuple(bbox[0]), tuple(bbox[2]), color, thickness)

    def draw_square(self, left_right, alg, color, thickness):
        bbox = self.get_bbox(alg)
        for rectified in left_right:
            cv2.rectangle(rectified, tuple(bbox[0]), tuple(bbox[2]), color, thickness)

    def abs_crop_square(self, left_right, i, j):
        bbox = self.get_abs_bbox(i, j)
        starts = np.min(bbox, axis=0).astype(int)
        stops = np.max(bbox, axis=0).astype(int) + 1
        bbox = bbox.reshape((1, -1, 1, 2)).astype(np.int32)
        out = []
        for rectified in left_right:
            out.append(rectified[starts[1]:stops[1],starts[0]:stops[0]])
        return out, bbox

    def crop_square(self, left_right, alg):
        i,j = self.alg2ij(alg)
        return self.abs_crop_square(left_right, i, j)


    def capture_raw(self):
        frame = None
        ret, frame = self.vid.read()
        if frame is None:
            raise ValueError("Can't read camera image")
        #frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = cv2.flip(frame, 0)
        frame = cv2.flip(frame, 1)
        return frame[:,:IM_WIDTH], frame[:,IM_WIDTH:]

    def capture_rectified(self):
        frames = self.capture_raw()

        rectified_pair = [None, None]
        for k, frame in enumerate(frames):
            rectified_pair[k] = cv2.warpPerspective(frame, self.perspective_matrices[k],
                                                    (IM_HEIGHT, IM_HEIGHT),
                                                    flags=cv2.INTER_LINEAR)
            
        return rectified_pair
        
    def imshow(self, name, frame):
        cv2.imshow(name, frame)

    def centerup(self):
        print("Center board in field of view.  Press 'q' to continue.")
        font = getattr(cv2, defaults.font_name)
        frames = self.capture_raw()


        while 1:
            frames = self.capture_raw()

            for i, frame in enumerate(frames):
                frame = cv2.putText(frame, f'Calibrating camera ...',
                                    (150,40), font, 
                                    1, RED, 1, cv2.LINE_AA)
                frame = cv2.putText(frame, f'... setup board,',
                                    (200,IM_HEIGHT//2 - 40), font, 
                                    1, RED, 1, cv2.LINE_AA)
                frame = cv2.putText(frame, f'press "q" when centered.',
                                    (100,IM_HEIGHT//2+40), font, 
                                    1, RED, 1, cv2.LINE_AA)
                pos = (IM_WIDTH // 2, IM_HEIGHT//2)
                frame = cv2.circle(frame,  pos, 10, RED, 2)

                if frame is None:
                    print("frame capture failed")
                    break
                self.imshow(f'Calibrate{i}', frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

    def findChessboardCorners(self, max_tries=2):
        print('Locating chessboard...')
        cornerss = [[], []]
        print(f'iter: {iter}/{max_tries}')
        frames = self.capture_raw()
        for k, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (7, 7))
            if ret:
                font = getattr(cv2, defaults.font_name)
                corners = corners.squeeze()
                sort = corners[:,0] + IM_HEIGHT * corners[:,1]
                corners = corners[np.argsort(sort)]
                corners = corners.reshape((7, -1, 2))
                for i, row in enumerate(corners):
                    row = row[np.argsort(row[:,0])]
                    corners[i] = row
                cornerss[k] = corners
        corners[0] = np.array(corners[0])
        corners[1] = np.array(corners[1])

        images = self.capture_raw()
        font = getattr(cv2, defaults.font_name)

        for k, image in enumerate(images):
            for i, row in enumerate(cornerss[k]): ## 7x
                for j, c in enumerate(row):   ## 7x
                    pos = tuple(c.astype(int))
                    cv2.circle(image,  pos, 10, (56, 123, 26), 4)

                    image = cv2.putText(image, f'{i}{j}', pos, font, 
                                        1, RED, 1, cv2.LINE_AA)
                    
            cx, cy = Util.plane(cornerss[k])
            _i = np.arange(7) 
            _j = np.array([-2, -1, 3, 4])
            j, i = Util.flatmeshgrid(_j, _i)
            xy = Util.eval_plane(cx, cy, i, j).reshape((7, 4, 2))
            #orig_corners = corners
            #corners[k] = np.column_stack([xy[:,:2], corners[k], xy[:,2:]])
            
            
##############################################################################
        if False:### double check corners?
            for k in range(2):
                pl.figure(k)
                xy = cornerss[k].reshape((-1, 2))
                #orig_xy = orig_corners[k].reshape((7, -1, 2))
                pl.plot(xy[:,0], xy[:,1], 'wo')
                #pl.plot(orig_xy[:,0], orig_xy[:,1], 'kx')
                for i, row in enumerate(corners): ## 7x
                    for j, c in enumerate(row):   ## 7x
                        pos = tuple(c.astype(int))
                        print(k, i, j, c)
                        #cv2.circle(images[k],  pos, 10, (56, 123, 26), 4)
                        #image = cv2.putText(images[k], f'{i}{j}', pos, font, 1, GREEN, 1, cv2.LINE_AA)
                pl.imshow(images[k])
            pl.show()
            input('here')
            sys.exit()
##############################################################################
        return cornerss

    def abs_square_occupied(self, left_right, i, j, thresh=50):
        lr, bbox = self.abs_crop_square(left_right, i, j)
        l, r = lr
        mask = cv2.threshold(cv2.absdiff(l, r), thresh, 255, cv2.THRESH_BINARY)[1]
        mask = mask.astype(bool)
        return np.sum(mask) > 50

    def square_occupied(self, left_right, alg):
        i, j = self.alg2ij(alg)
        return self.abs_square_occupied(left_right, i, j)

    def calibrate(self):
        # Capture the video frame
        self.centerup()
        cornerss = self.findChessboardCorners(1)

        ### extend 7x7 corners found in calibration to edge of board
        ### using a polynomial fit
        Ms = [None, None]
        coeffs = [None, None]
        for k, corners in enumerate(cornerss):
            ij = np.empty((49, 2))
            xy = np.empty((49, 2))
            for i, row in enumerate(corners):
                for j, pos in enumerate(row):
                    pos = tuple(pos.astype(int))
                    ij[i * 7 + j] = i, j
                    xy[i * 7 + j] = pos

            coeff = board_map.fit(ij, xy)
            coeffs[k] = coeff
            # find baord edge
            coords = np.array([[-1.5, -1.5],
                               [-1.5,  7.5],
                               [ 7.5,  7.5],
                               [ 7.5, -1.5]])[::-1]
            bbox = board_map.predict(coords, coeff).astype(int)
            input_pts = np.float32(np.roll(bbox, 0, axis=0))
            output_pts = np.float32([[0, 0],
                                     [0, IM_HEIGHT - 1],
                                     [IM_HEIGHT - 1, IM_HEIGHT - 1],
                                     [IM_HEIGHT - 1, 0]])
            M = cv2.getPerspectiveTransform(input_pts,output_pts)
            Ms[k] = M
        font = getattr(cv2, defaults.font_name)
        while True:
            frames = self.capture_raw()
            for k, frame in enumerate(frames):
                rectified = cv2.warpPerspective(frame, Ms[k], (IM_HEIGHT, IM_HEIGHT),
                                                flags=cv2.INTER_LINEAR)        
                rectified = cv2.putText(rectified, f'Press "q" to continue.',
                                        (10,IM_HEIGHT//2), font, 
                                        1, RED, 1, cv2.LINE_AA)
                rectified = cv2.putText(rectified, f'Press "x" to redo.',
                                        (10,IM_HEIGHT//2 + 40), font, 
                                        1, RED, 1, cv2.LINE_AA)
                self.imshow(f'Calibrate{k}', rectified)
            key = chr(cv2.waitKey(1) & 0xFF)
            if key == 'q':
                break
            if key == 'x':
                self.calibrate()
                break
            perspective_matrix, ij_coeff = M, coeff
        perspective_matrices = Ms
        np.savez(self.cal_npz, perspective_matrix=perspective_matrices)
        print('wrote', self.cal_npz)
        print('Calibaration complete.')
        cv2.destroyAllWindows()
        self.perspective_matrix = perspective_matrix
        return Ms, coeffs

    def __del__(self):
        print('release video')
        #self.vid.release()

    def abs_piece_color(self, left_right, i, j):
        '''
        i, j -- absolute coords
        sq_left_right -- (sq_left, sq_right)
        '''
        lr, bbox = self.abs_crop_square(left_right, i, j)
        l, r = lr
        mask = cv2.threshold(cv2.absdiff(l, r), 25, 255, cv2.THRESH_BINARY)[1]
        mask = mask.astype(bool)
        dl = np.where(mask, l, np.nan)
        #cv2.imshow(f'l{i}{j}', l)
        #cv2.imshow(f'dl{i}{j}', dl)
        out = np.nanmean(dl.reshape((-1, 3)), axis=0)
        dw = np.linalg.norm(out - white)
        db = np.linalg.norm(out - black)
        if dw > db:
            out = chess.BLACK
        else:
            out = chess.WHITE
        return out
        

black = np.array([ 92.18406593, 104.33333333, 168.52941176])
white = np.array([137.80169972, 125.58855586, 118.53432836])
if __name__ == '__main__' and n_cam > 0:
    desc = 'Chess camera utility library'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-c','--calibrate',
                        help='Calibrate board area',
                        required=False, default=False)
    parser.add_argument('-d','--display',
                        help='Display board area',
                        required=False, default="True")
    args = parser.parse_args()

    import matplotlib; matplotlib.interactive(True)
    cc = ChessCam(camera_number = 2)
    if args.calibrate:
        cc.calibrate()
        sys.exit()
    n = 0
    aa = np.zeros(3)
    bb = np.zeros(3)
    while 1:
        left_right = cc.capture_rectified()
        for i in range(1, 9):
            for j in range(1, 9):
                if cc.abs_square_occupied(left_right, i, j):
                    color = 'BW'[cc.abs_piece_color(left_right, i, j)]
                    print(color, end=' ')
                else:
                    print(' ', end=' ')
            print()
        print()
        print()
        #res = cv2.bitwise_and(l, l, mask=mask)
        #cv2.imshow('left', left)
        #cv2.imshow('right', right)
        #if cv2.waitKey(1) == ord('q'):
        #    break
        break
