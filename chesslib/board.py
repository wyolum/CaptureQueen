import chess
from itertools import groupby
from copy import deepcopy

from . import pieces
import re

class ChessError(Exception): pass
class InvalidCoord(ChessError): pass
class InvalidColor(ChessError): pass
class InvalidMove(ChessError): pass
class Check(ChessError): pass
class CheckMate(ChessError): pass
class Draw(ChessError): pass
class NotYourTurn(ChessError): pass

FEN_STARTING = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
RANK_REGEX = re.compile(r"^[A-Z][1-8]$")

class Board(dict):
    '''
       Board

       A simple chessboard class

       TODO:

        * PGN export
        * En passant (Done TJS)
        * Castling (Done TJS)
        * Promoting pawns (Done TJS)
        * 3-time repition (Done TJS)
        * Fifty-move rule
        * Take-backs
        * row/column lables
        * captured piece imbalance (show how many pawns pieces player is up)
    '''

    axis_y = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H')
    axis_x = tuple(range(1,9)) # (1,2,3,...8)

    captured_pieces = { 'white': [], 'black': [] }
    player_turn = None
    castling = '-'
    en_passant = '-'
    halfmove_clock = 0
    fullmove_number = 1
    def __init__(self, fen=None):
        if fen is None:
            fen = FEN_STARTING
        self.load(fen)
        self.last_move = None ## to support en passent
        self.positions = [None]
        self.board = chess.Board(fen)
        
    def __getitem__(self, coord):
        if isinstance(coord, str):
            coord = coord.upper()
            if not re.match(RANK_REGEX, coord.upper()): raise KeyError
        elif isinstance(coord, tuple):
            coord = self.letter_notation(coord)
        try:
            return super(Board, self).__getitem__(coord)
        except KeyError:
            return None

    def move(self, p1, p2, promote='q'):
        move = chess.Move.from_uci("g1f3")
        print('p1:', p1, 'p2:', p2)

    def all_possible_moves(self, color):
        '''
            Return a list of `color`'s possible moves.
            Does not check for check.
        '''
        if(color not in ("black", "white")): raise InvalidColor
        result = []
        for coord in list(self.keys()):
            if (self[coord] is not None) and self[coord].color == color:
                moves = self[coord].possible_moves(coord)
                if moves: result += moves
        return result

    def occupied(self, color):
        '''
            Return a list of coordinates occupied by `color`
        '''
        result = []
        if(color not in ("black", "white")): raise InvalidColor

        for coord in self:
            if self[coord].color == color:
                result.append(coord)
        return result

    def is_king(self, piece):
        return isinstance(piece, pieces.King)
    
    def get_king_position(self, color):
        for pos in list(self.keys()):
            if self.is_king(self[pos]) and self[pos].color == color:
                return pos

    def get_king(self, color):
        if(color not in ("black", "white")): raise InvalidColor
        return self[self.get_king_position(color)]

    def get_other(self, color):
        if color == 'white':
            other = 'black'
        else:
            other = 'white'
        return other
    
    def is_in_check(self, color):
        return self.board.is_check()
        if(color not in ("black", "white")): raise InvalidColor
        king = self.get_king(color)
        other = self.get_other(color)
        return king in list(map(self.__getitem__, self.all_possible_moves(other)))

    def letter_notation(self,coord):
        if not self.is_in_bounds(coord): return
        try:
            return self.axis_y[coord[1]] + str(self.axis_x[coord[0]])
        except IndexError:
            raise InvalidCoord

    def number_notation(self, coord):
        coord = coord.upper()
        out = int(coord[1])-1, self.axis_y.index(coord[0])
        return out

    def is_in_bounds(self, coord):
        if coord[1] < 0 or coord[1] > 7 or\
           coord[0] < 0 or coord[0] > 7:
            return False
        else: return True
    def clear(self):
        dict.clear(self)
        self.poistions = [None]

    def validate_fen(self, fen):
        return len(fen.strip().split()) == 6
    
    def load(self, fen):
        '''
            Import state from FEN notation
        '''
        self.board = chess.Board(fen)
        self.clear()
        # Split data
        fen = fen.split(' ')
        # Expand blanks
        def expand(match): return ' ' * int(match.group(0))

        fen[0] = re.compile(r'\d').sub(expand, fen[0])
        self.positions = [None]
        for x, row in enumerate(fen[0].split('/')):
            for y, letter in enumerate(row):
                if letter == ' ': continue
                coord = self.letter_notation((7-x,y))
                self[coord] = pieces.piece(letter)
                self[coord].place(self)

        if fen[1] == 'w': self.player_turn = 'white'
        else: self.player_turn = 'black'

        self.castling = fen[2]
        self.en_passant = fen[3]
        self.halfmove_clock = int(fen[4])
        self.fullmove_number = int(fen[5])
        
    def can_en_passent(self):
        out = '-'
        if self.last_move and abs(int(self.last_move[1][1]) - int(self.last_move[0][1])) == 2:
            if self.is_pawn(self[self.last_move[1]]): ### yes we can
                out = self.last_move[1][0].lower()
                if self.last_move[1][1] == '4':
                    out += '3'
                else:
                    out += '6'
        return out
        
    def export(self):
        '''
            Export state to FEN notation
        '''
        return self.board.fen()
