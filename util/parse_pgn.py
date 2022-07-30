import chess
import chess.pgn
import re

fn = 'game.pgn'

f = open(fn)
game = chess.pgn.read_game(f)
board = game.board()

for move in game.mainline_moves():
    board.push_uci(move.uci())
    print(board.fen())
