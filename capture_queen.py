from collections import defaultdict
import chess
import os
import time
import tkinter
from chesslib import board
from chesslib import gui_tkinter
import subprocess
import argparse
from stockfish import Stockfish

def exit():
    if __after_job:
        r.after_cancel(__after_job)
    
parser = argparse.ArgumentParser(
    description='Capture a game of chess over the board')
parser.add_argument("--depth", type=int,
                    help='stockfish search depth',
                    default=15)
args = parser.parse_args()
current_proc = None
depth = args.depth
n_player = 2


__engine = [None]
max_depth = 16
min_depth = 10
def getEngine():
    if __engine[0] is None:
        __engine[0] = Stockfish(path="/usr/games/stockfish", depth=depth)
    return __engine[0]


__last_fen = ''
__after_job = False
__last_best = None
__current_depth = None
__status = None
__single_move = None

def check_single_move(old_fen, new_fen):
    '''
    return algebric notation for move if new_fen is a single move from old_fen
    '''
    if old_fen:
        b1 = chess.Board(old_fen)
        for move in b1.legal_moves:
            move = move.uci()
            b1.push_uci(move)
            if b1.fen() == new_fen:
                return move
            b1.pop()

def zero():
    return 0
material_values = defaultdict(zero)
material_values.update({'p':-1, 'P':1,
                        'n':-3, 'N':3,
                        'b':-3, 'B':3,
                        'r':-5, 'R':5,
                        'q':-9, 'Q':9,
})
def get_material(fen):
    if fen:
        pieces = fen.split()[0]
        out = sum([material_values[c] for c in pieces])
    else:
        out = 0
    return out

def go():
    global __last_fen, __after_job, __last_best, __current_depth, __status
    global __single_move
    
    engine = getEngine()
    if os.path.exists('.fen'):
        fen = open('.fen').read().strip()
        if fen != __last_fen and len(fen) > 10:
            __single_move = check_single_move(__last_fen, fen)
            gui.update(fen)
            __last_fen = fen
            engine.set_fen_position(fen)
            __current_depth = min_depth
            engine.set_depth(min_depth) ## keep it resposive
            
    eval = engine.get_evaluation()
    centipawn= eval['value']
    gui.set_eval(centipawn)
    best = engine.get_best_move()
    material = get_material(fen)
    status = f'Best: {best}, Eval:{centipawn/100:+.2f}, Depth:{__current_depth}'
    status += f' Material: {material:+2d}'

    gui.refresh()
    gui.clear_arrows()
    gui.label_status["text"] = status
    __status = status

    gui.draw_arrow(best[:2], best[2:], '#0000FF80')

    gui.clear_arrows()
    gui.draw_arrow(best[:2], best[2:], '#0000FF80')
    __last_best = best
    if __single_move:
        start = __single_move[:2]
        stop = __single_move[2:]
        gui.draw_arrow(__single_move[:2],
                       __single_move[2:],
                       color='#B0CB0260')
    if __current_depth < max_depth:
        __current_depth += 1
        engine.set_depth(__current_depth)
    __after_job = r.after(1, go)
    return

game = board.Board()
r = tkinter.Tk()
#r.attributes('-type', 'dock')
#r.geometry('+0+0') ## place in touch screen
r.after(100, go)
gui = gui_tkinter.BoardGuiTk(r, game, engine=getEngine())
gui.pack()
gui.draw_pieces()
input('>>')
exit()
#r.mainloop()





