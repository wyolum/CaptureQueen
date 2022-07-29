import time
from tkinter import *
from chesslib import board
from chesslib import gui_tkinter
import subprocess
import argparse
from stockfish import Stockfish

parser = argparse.ArgumentParser(description='Capture a game of chess over the board')
parser.add_argument("--depth", type=int,
                    help='stockfish search depth',
                    default=10)
args = parser.parse_args()
current_proc = None
depth = args.depth
n_player = 2


__engine = [None]
def getEngine():
    if __engine[0] is None:
        __engine[0] = Stockfish(path="/usr/games/stockfish", depth=depth)
    return __engine[0]

moves = []
def isEnPassent(move, piece):
    out = False
    ## pawn to blank space
    if piece.abbriviation.lower() != 'p' or game[move[2:4]] is not None:
        out = False
    elif move[1] == '5' and move[3] == '6' and move[0] != move[2]: ## white diag
        out = move[2] + '5'
    elif move[1] == '4' and move[3] == '3' and move[0] != move[2]: ## black diag
        out = move[2] + '3'
    return out

def isCastle(move, piece):
    '''return move for castle if it is else False'''
    move = move.upper()
    if piece.abbriviation.lower() != 'k':
        out = False
    elif move == 'E1G1' and piece.color == 'white':
        out = 'H1F1'
    elif move == 'E1C1' and piece.color == 'white':
        out = 'A1D1'
    elif move == 'E8G8' and piece.color == 'black':
        out = 'H8F8'
    elif move == 'E8C8' and piece.color == 'black':
        out = 'A8D8'
    else:
        out = False
    return out

def go():
    if n_player == 2:
        return
    elif n_player == 1 and game.export().split()[1] == 'w':
        r.after(1000, go)
        return
    engine = getEngine()
    fen = game.export()
    command = 'position fen %s\n' % fen
    # print '<<<', command
    move = engine.get_best_move().upper()
    
    piece = gui.chessboard[move[:2]]
    if len(move) > 4:
        promote = move[4]
    else:
        promote = None
    gui.move(move[:2], move[2:4], promote=promote)
    engine.make_moves_from_current_position([move.lower()])
    enemy = gui.chessboard.get_enemy(piece.color)
    moves.append(move)
    # print(engine.get_evaluation())
    # print move,
    if piece.color == 'black':
        gui.chessboard.fullmove_number += 1
        # print ''
    gui.chessboard.halfmove_clock +=1
    gui.chessboard.player_turn = enemy
    gui.refresh()
    gui.draw_pieces()
    gui.redraw_square(gui.from_square, 'tan1')
    gui.redraw_square(gui.to_square, 'tan1')
    if gui.chessboard.is_in_check(enemy):
        p = gui.chessboard.get_king_position(enemy)
        p = gui.chessboard.number_notation(p)
        gui.redraw_square(p, 'red')
    r.after(100, go)

def on_move(move):
    engine = getEngine()
    if not engine.is_move_correct(move):
        return
    print('capture_queen::on_move():', move)
    for m in engine.get_top_moves(3):
        if m['Mate']:
            mate = 'Mate'
        else:
            mate = ''
        print(m['Move'], m['Centipawn'], mate)
    print(engine.get_board_visual())
    engine.make_moves_from_current_position([move.lower()])

game = board.Board()
r = Tk()
r.after(1000, go)
gui = gui_tkinter.BoardGuiTk(r, game, engine=getEngine())
gui.pack()
gui.draw_pieces()

r.mainloop()

