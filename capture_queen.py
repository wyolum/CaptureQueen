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


last_fen = ''
__after_job = False
__last_best = None
__current_depth = None
__status = None
def go():
    global last_fen, __after_job, __last_best, __current_depth, __status
    engine = getEngine()
    if os.path.exists('.fen'):
        fen = open('.fen').read().strip()
        if fen != last_fen and len(fen) > 10:
            print(fen)
            gui.update(fen)
            last_fen = fen
            #gui.draw_arrow('c6', 'd8')
            #gui.draw_arrow('e7', 'd8', '#FF0000A0')
            engine.set_fen_position(fen)
            __current_depth = min_depth
            engine.set_depth(min_depth) ## keep it resposive
    eval = engine.get_evaluation()
    centipawn= eval['value']
    gui.set_eval(centipawn)
    best = engine.get_best_move()
    status = f'Best: {best}, Eval:{centipawn/100:.2f}, Depth:{__current_depth}'
    if __status != status:
        gui.label_status["text"] = status
        __status = status
        gui.refresh()
        gui.clear_arrows()
        gui.draw_arrow(best[:2], best[2:])
    if __last_best != best:
        gui.clear_arrows()
        gui.draw_arrow(best[:2], best[2:])
        __last_best = best
    if __current_depth < max_depth:
        __current_depth += 1
        engine.set_depth(__current_depth)
    __after_job = r.after(100, go)
    return

game = board.Board()
r = tkinter.Tk()
r.attributes('-type', 'dock')
r.geometry('+0+0')
r.after(1000, go)
gui = gui_tkinter.BoardGuiTk(r, game, engine=getEngine())
gui.pack()
gui.draw_pieces()
input('>>')
exit()
#r.mainloop()





