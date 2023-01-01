import pylab as pl
import numpy as np
import chess
import cv2

from colors import RED, GREEN, BLUE, CYAN, PURPLE, WHITE, BLACK, GRAY
import defaults
import Util
from Util import linfit, intersect
import board_map
IM_WIDTH = 640
IM_HEIGHT = 480

DELTA = IM_HEIGHT / 9
BBOX = (np.array([
    [-0.5,  7.5],
    [-0.5, -0.5],
    [ 7.5, -0.5],
    [ 7.5,  7.5],
]) + 1) * DELTA


class ChessCam:
    ### flip_board
    ### If True, the right side of board from camera view point is toward the
    ### bottom of the board.
    ### If False, the left side of the board is toward the bottom

    def __init__(self, flip_board=False, side=chess.WHITE,
                 cal_npz='perspective_matrix.npz', camera_number=0):
        self.flip_board = flip_board
        self.side = side
        self.perspective_matrix = np.load(cal_npz)['perspective_matrix']
        self.vid = cv2.VideoCapture(camera_number)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, IM_WIDTH)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, IM_HEIGHT)
        self.vid.set(cv2.CAP_PROP_EXPOSURE, 15)
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
                           [ 1, -1]]) / 6 + np.array([i, j]) ### to make bigger use  / 3, (/2 = full sq)
        bbox = coords * DELTA
        return bbox.astype(int)

    def get_bbox(self, alg):
        i = ord(alg[0]) - ord('a') + 1
        j = 9 - int(alg[1])
        if self.side == chess.BLACK:
            j = 9 - j
            i = 9 - i
        bbox = get_abs_bbox(i, j)
        return bbox

    def draw_abs_square(self, rectified, i, j, color, thickness):
        bbox = self.get_abs_bbox(i, j)
        cv2.rectangle(rectified, tuple(bbox[0]), tuple(bbox[2]), color, thickness)

    def draw_square(self, rectified, alg, color, thickness):
        bbox = self.get_bbox(alg)
        cv2.rectangle(rectified, tuple(bbox[0]), tuple(bbox[2]), color, thickness)

    def crop_abs_square(self, rectified, i, j):
        bbox = self.get_abs_bbox(i, j)
        starts = np.min(bbox, axis=0).astype(int)
        stops = np.max(bbox, axis=0).astype(int) + 1
        bbox = bbox.reshape((1, -1, 1, 2)).astype(np.int32)
        out = rectified[starts[1]:stops[1],starts[0]:stops[0]], bbox
        return out

    def crop_square(self, rectified, alg):
        bbox = self.get_bbox(alg)
        starts = np.min(bbox, axis=0).astype(int)
        stops = np.max(bbox, axis=0).astype(int) + 1
        bbox = bbox.reshape((1, -1, 1, 2)).astype(np.int32)
        out = rectified[starts[1]:stops[1],starts[0]:stops[0]], bbox
        return out


    def capture_rectified(self):
        ret, frame = self.vid.read() 
        rectified = cv2.warpPerspective(frame, self.perspective_matrix,
                                        (IM_HEIGHT, IM_HEIGHT),
                                        flags=cv2.INTER_LINEAR)
        return rectified
        
    def imshow(self, name, frame):
        cv2.imshow(name, frame)

    def centerup(self):
        print("Center board in field of view.  Press 'q' to continue.")
        font = getattr(cv2, defaults.font_name)
        while 1:
            ret, frame = self.vid.read()
            frame = cv2.putText(frame, f'Calibrating camera ...',
                                (150,40), font, 
                                1, RED, 1, cv2.LINE_AA)
            frame = cv2.putText(frame, f'... clear board,',
                                (200,IM_HEIGHT//2 - 40), font, 
                                1, RED, 1, cv2.LINE_AA)
            frame = cv2.putText(frame, f'press "q" when centered.',
                                (100,IM_HEIGHT//2), font, 
                                1, RED, 1, cv2.LINE_AA)
            if frame is None:
                print("frame capture failed")
                break
            self.imshow('Calibrate', frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

    def findChessboardCorners(self, n_ave=1, max_tries=20):
        print('Locating chessboard...')
        all_corners = []
        iter = 0
        while len(all_corners) < n_ave and iter < max_tries:
            print(f'iter: {iter}/{max_tries} {len(all_corners)}/{n_ave}')
            ret, frame = self.vid.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #ret, corners = cv2.findChessboardCorners(gray, (7, 7))
            ret, corners = cv2.findChessboardCorners(gray, (7, 3))
            if ret:
                font = getattr(cv2, defaults.font_name)
                corners = corners.squeeze()
                sort = corners[:,0] + IM_HEIGHT * corners[:,1]
                corners = corners[np.argsort(sort)]
                corners = corners.reshape((7, 3, 2))
                for i, row in enumerate(corners):
                    row = row[np.argsort(row[:,0])]
                    corners[i] = row
                all_corners.append(corners)
            iter += 1
        all_corners = np.array(all_corners)
        corners = np.mean(all_corners, axis=0)

        ret, image = self.vid.read()
        font = getattr(cv2, defaults.font_name)
        for i, row in enumerate(corners): ## 7x
            for j, c in enumerate(row):   ## 3x
                pos = tuple(c.astype(int))
                cv2.circle(image,  pos, 10, (56, 123, 26), 4)

                image = cv2.putText(image, f'{i}{j}', pos, font, 
                                    1, RED, 1, cv2.LINE_AA)

        cx, cy = Util.plane(corners)
        _i = np.arange(7) 
        _j = np.array([-2, -1, 3, 4])
        j, i = Util.flatmeshgrid(_j, _i)
        xy = Util.eval_plane(cx, cy, i, j).reshape((7, 4, 2))
        corners = np.column_stack([xy[:,:2], corners, xy[:,2:]])

##############################################################################
#        xy = corners.reshape((-1, 2))
#        pl.plot(xy[:,0], xy[:,1], 'wo')
#        for i, row in enumerate(corners): ## 7x
#            for j, c in enumerate(row):   ## 7x
#                pos = tuple(c.astype(int))
#                cv2.circle(image,  pos, 10, (56, 123, 26), 4)
#
#                image = cv2.putText(image, f'{i}{j}', pos, font, 
#                                    1, GREEN, 1, cv2.LINE_AA)
#        pl.imshow(image)
#        input('here')
##############################################################################
        return corners
    
    def calibrate(self):
        # Capture the video frame
        self.centerup()
        corners = self.findChessboardCorners(1)

        ### extend 7x7 corners found in calibration to edge of board
        ### using a polynomial fit

        ij = np.empty((49, 2))
        xy = np.empty((49, 2))
        for i, row in enumerate(corners):
            for j, pos in enumerate(row):
                pos = tuple(pos.astype(int))
                ij[i * 7 + j] = i, j
                xy[i * 7 + j] = pos

        coeff = board_map.fit(ij, xy)

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
        font = getattr(cv2, defaults.font_name)
        while True:
            ret, frame = self.vid.read()
            rectified = cv2.warpPerspective(frame, M, (IM_HEIGHT, IM_HEIGHT),
                                       flags=cv2.INTER_LINEAR)        

            rectified = cv2.putText(rectified, f'Press "q" to continue.',
                                    (10,IM_HEIGHT//2), font, 
                                    1, RED, 1, cv2.LINE_AA)
            rectified = cv2.putText(rectified, f'Press "x" to redo.',
                                    (10,IM_HEIGHT//2 + 40), font, 
                                    1, RED, 1, cv2.LINE_AA)
            self.imshow('Calibrate', rectified)
            key = chr(cv2.waitKey(1) & 0xFF)
            if key == 'q':
                break
            if key == 'x':
                self.calibrate()
                break
        perspective_matrix, ij_coeff = M, coeff
        np.savez(self.cal_npz, perspective_matrix=perspective_matrix)
        print('wrote', self.cal_npz)
        print('Calibaration complete.')
        cv2.destroyAllWindows()
        return M, coeff
    
if __name__ == '__main__':
    import matplotlib; matplotlib.interactive(True)
    cc = ChessCam()
    #cc.calibrate()
    #input('here')
    #here
    rect = cc.capture_rectified()
    jpg = 'colortest.jpg'
    cv2.imwrite(jpg, rect)
    print('wrote', jpg)
    if False:
        occupied = np.zeros((8, 8), bool)
        occupied[:7,0] = True
        occupied[:7,1] = True
        occupied[:7,6] = True
        occupied[:7,7] = True
        occupied[6,:] = True

        pl.figure(1, figsize=(5, 5)); pl.clf()
        fig, ax = pl.subplots(8,8, num=1)

        scores = {True:[], False:[]}
        maxes = {True:[], False:[]}
        nbin = 256
        for i in range(8):
            for j in range(8):
                sq, bbox = cc.crop_abs_square(rect, i+1, j+1)
                hist = cv2.calcHist([sq],[0],None,[nbin],[0,nbin])
                hist = hist[1:] + hist[:-1]
                x = np.arange(len(hist))/len(hist) * sq.shape[0]
                y = (1 - hist / np.max(hist)) * sq.shape[1]
                thresh = 10
                ax[i][j].imshow(sq)
                ax[i][j].plot(x, y)
                ax[i][j].axis('off')
                ax[i][j].text(15, 5, np.sum(hist > 1), color='r')
                ax[i][j].text(15, 10, np.sum(hist > 2), color='g')
                ax[i][j].text(15, 15, np.max(hist), color='b')
                if occupied[i][j]:
                    ax[i][j].text(5, 15, 'X', color='w')
                score = [np.sum(hist > t) for t in [1, 2]]
                scores[occupied[i][j]].append(score)
                maxes[occupied[i][j]].append(np.max(hist))



        pl.figure(2, figsize=(12, 12)); pl.clf()
        fig, ax = pl.subplots(1, num=2)
        markers = ['.', 'x']
        for i in range(2):
            s = np.array(scores[i])
            ax.plot(s[:,0], maxes[i], markers[i])
            #ax.plot(s[:,1], maxes[i], markers[i], markersize=5)
        input('huh?>')
