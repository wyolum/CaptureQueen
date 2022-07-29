import os
import glob
from . import board
from . import pieces
import tkinter as tk
from PIL import Image, ImageTk

color1 ='white'
color2 = 'grey'

def get_color_from_coords(coords):
    return [color1, color2][(coords[0] - coords[1]) % 2]

class BoardGuiTk(tk.Frame):
    pieces = {}
    selected = None
    selected_piece = None
    hilighted = None    
    icons = {}

    rows = 8
    columns = 8

    def ask_piece(self, color):
        if not self.prompting:
            top = tk.Toplevel()
            codes = 'QRBN'
            def on_click(event):
                x = event.x // self.square_size
                if 0 <= x and x < len(codes):
                    self.promote_code = codes[x]
                    top.destroy()

            canvas = tk.Canvas(top,
                               width=len(codes) * self.square_size,
                               height=self.square_size,
                               background='grey')
            canvas.bind('<Button-1>', on_click)
            canvas.pack(side="top", fill="both", anchor="c", expand=True)
            x = -self.square_size / 2
            y = self.square_size / 2
            for i, code in enumerate(codes):
                x += self.square_size
                canvas.create_rectangle(i * self.square_size, 0,
                                        (i + 1) * self.square_size, self.square_size,
                                        outline="black",
                                        fill=[color1, color2][i % 2],
                                        tags="square")
                filename = "img/%s%s.png" % (color, code.lower())
                piecename = "%s%s%s" % (code.lower, x, y)
                canvas.create_image(x, y, image=self.icons[filename], tags=("piece",), anchor="c")
            self.prompting = True
            parent_geom = self.parent.geometry()
            wh, x, y = parent_geom.split('+')
            x = int(x)
            y = int(y)
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            w, h = wh.split('x')
            w = int(w)
            h = int(h)

            geom = "%dx%d%+d%+d" % (len(codes) * self.square_size,
                                    self.square_size, x, y)
            top.geometry(geom)
            self.wait_window(top)
            self.prompting = False
    @property
    def canvas_size(self):
        return (self.columns * self.square_size,
                self.rows * self.square_size)

    def __init__(self, parent, chessboard, square_size=64, engine=None):
        self.color1 = color1
        self.color2 = color2
        self.chessboard = chessboard
        self.square_size = square_size
        self.parent = parent
        self.from_square = None
        self.to_square = None
        self.prompting = False
        self.engine = engine

        canvas_width = self.columns * square_size
        canvas_height = self.rows * square_size

        tk.Frame.__init__(self, parent)

        self.canvas = tk.Canvas(self, width=canvas_width, height=canvas_height, background="grey")
        self.canvas.pack(side="top", fill="both", anchor="c", expand=True)

        self.canvas.bind("<Configure>", self.refresh)
        self.canvas.bind("<Button-1>", self.click)

        self.statusbar = tk.Frame(self, height=64)
        self.button_quit = tk.Button(self.statusbar, text="New", fg="black", command=self.reset)
        self.button_quit.pack(side=tk.LEFT, in_=self.statusbar)

        self.button_save = tk.Button(self.statusbar, text="Save", fg="black", command=self.chessboard.save_to_file)
        self.button_save.pack(side=tk.LEFT, in_=self.statusbar)

        self.label_status = tk.Label(self.statusbar, text="   White's turn  ", fg="black")
        self.label_status.pack(side=tk.LEFT, expand=0, in_=self.statusbar)

        self.button_quit = tk.Button(self.statusbar, text="Quit", fg="black", command=self.parent.destroy)
        self.button_quit.pack(side=tk.RIGHT, in_=self.statusbar)
        self.statusbar.pack(expand=False, fill="x", side='bottom')


    def redraw_square(self, coord, color=None):
        row, col = coord
        if color is None:
            color = get_color_from_coords(coord)
        x1 = (col * self.square_size)
        y1 = ((7-row) * self.square_size)
        x2 = x1 + self.square_size
        y2 = y1 + self.square_size
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="black", fill=color, tags="square")

        ## see if we need to redraw a piece
        piece = self.chessboard[(row, col)]
        if piece is not None:
            self.draw_piece(piece, row, col)
    
    def click(self, event):

        # Figure out which square we've clicked
        col_size = row_size = event.widget.master.square_size

        current_column = event.x // col_size
        current_row = 7 - (event.y // row_size)
        
        position = self.chessboard.letter_notation((current_row, current_column))
        piece = self.chessboard[position]
        if self.from_square is not None: ## undraw from
            self.redraw_square(self.from_square)
            self.redraw_square(self.to_square)

        if self.selected_piece:
            piece = self.selected_piece[1]
            self.move(piece, position)
            self.selected_piece = None
            self.hilighted = None
            self.pieces = {}
            self.refresh()
            self.draw_pieces()
        else:
            self.hilighted = None
        self.hilight(position)
        self.refresh()

        if self.from_square is not None:
            self.redraw_square(self.from_square, 'tan1')
            self.redraw_square(self.to_square, 'tan1')
        if self.hilighted is not None:
            for square in self.hilighted:
                self.redraw_square(square, 'spring green')
        enemy = self.chessboard.player_turn
        if self.chessboard.is_in_check(enemy):
            algebraic_pos = self.chessboard.get_king_position(enemy)
            enemy_king_loc = self.chessboard.number_notation(algebraic_pos)
            self.redraw_square(enemy_king_loc, 'red')
            

    def move(self, p1, p2, promote=None):
        '''
        maybe a move, or just second click
        '''
        piece = self.chessboard[p1]
        enemy = self.chessboard.get_enemy(piece.color)
        dest_piece = self.chessboard[p2]
        if dest_piece is None or dest_piece.color != piece.color:
            try:
                if self.from_square:
                    self.redraw_square(self.from_square)
                if self.to_square:
                    self.redraw_square(self.to_square)
                if isinstance(piece, pieces.Pawn) and p2[1] in '18':
                    ### promotion!
                    if promote is None:
                        self.ask_piece(piece.color)
                        promote = self.promote_code
                else:
                    promote = ''
                if self.engine is not None:
                    uci = f'{p1.lower()}{p2.lower()}{promote}'
                    self.engine.make_moves_from_current_position([uci])
                    print(self.engine.get_fen_position())
                self.chessboard.move(p1, p2, promote=promote)
                self.from_square = self.chessboard.number_notation(p1)
                self.to_square = self.chessboard.number_notation(p2)
                self.redraw_square(self.from_square, 'tan1')
                self.redraw_square(self.to_square, 'tan1')

                if self.chessboard.is_in_check(enemy):
                    algebraic_pos = self.chessboard.get_king_position(enemy)
                    enemy_king_loc = self.chessboard.number_notation(algebraic_pos)
                    self.redraw_square(enemy_king_loc, 'red')
                
            except board.InvalidMove as error:
                self.hilighted = []
            except board.ChessError as error:
                print('ChessError', error.__class__.__name__)
                self.label_status["text"] = error.__class__.__name__
                self.refresh()
                raise
            else:
                self.label_status["text"] = " " + piece.color.capitalize() +": "+ p1 + p2
            #if self.on_move is not None:
            #    move = f'{p1}{p1}'.lower()
            #    self.on_move(move)
                

    def hilight(self, pos):
        piece = self.chessboard[pos]
        if piece is not None and (piece.color == self.chessboard.player_turn):
            self.selected_piece = (self.chessboard[pos], pos)
            possible_moves = self.chessboard[pos].possible_moves(pos)
            possible_moves = [m for m in possible_moves if
                              not self.chessboard.is_in_check_after_move(pos, m)]
            self.hilighted = list(map(self.chessboard.number_notation, possible_moves))

    def addpiece(self, name, image, row=0, column=0):
        '''Add a piece to the playing board'''
        # self.canvas.create_image(0,0, image=image, tags=(name, "piece"), anchor="c")
        x = ((column + .5) * self.square_size)
        y = ((7-(row - .5)) * self.square_size)

        self.canvas.create_image(x, y, image=image, tags=(name, "piece"), anchor="c")
        self.placepiece(name, row, column)

    def placepiece(self, name, row, column):
        '''Place a piece at the given row/column'''
        self.pieces[name] = (row, column)
        x0 = (column * self.square_size) + int(self.square_size/2)
        y0 = ((7-row) * self.square_size) + int(self.square_size/2)
        self.canvas.coords(name, x0, y0)

    def refresh(self, event={}):
        '''Redraw the board'''
        if event:
            xsize = int((event.width-1) / self.columns)
            ysize = int((event.height-1) / self.rows)
            self.square_size = min(xsize, ysize)

        self.canvas.delete("square")
        color = self.color2
        for row in range(self.rows):
            color = self.color1 if color == self.color2 else self.color2
            for col in range(self.columns):
                x1 = (col * self.square_size)
                y1 = ((7-row) * self.square_size)
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                if (self.selected is not None) and (row, col) == self.selected:
                    self.redraw_square((row, col), 'orange')
                elif(self.hilighted is not None and (row, col) in self.hilighted):
                    self.redraw_square((row, col), 'spring green')
                else:
                    self.redraw_square((row, col), color)
                color = self.color1 if color == self.color2 else self.color2
        for name in self.pieces:
            self.placepiece(name, self.pieces[name][0], self.pieces[name][1])
        self.canvas.tag_raise("piece")
        self.canvas.tag_lower("square")

    def draw_piece(self, piece, row, col):
        x = (col * self.square_size)
        y = ((7-row) * self.square_size)
        filename = "img/%s%s.png" % (piece.color, piece.abbriviation.lower())
        piecename = "%s%s%s" % (piece.abbriviation, x, y)
        
        if(filename not in self.icons):
            self.icons[filename] = ImageTk.PhotoImage(file=filename, width=32, height=32)
        self.addpiece(piecename, self.icons[filename], row, col)
        #self.placepiece(piecename, row, col)                    

    def draw_pieces(self):
        self.canvas.delete("piece")
        
        for coord, piece in self.chessboard.items():
            if piece is not None:
                x,y = self.chessboard.number_notation(coord)
                self.draw_piece(piece, x, y)

    def reset(self):
        self.chessboard.load(board.FEN_STARTING)
        self.refresh()
        self.draw_pieces()
        self.refresh()

def display(chessboard):
    root = tk.Tk()
    root.title("Simple Python Chess")

    gui = BoardGuiTk(root, chessboard)
    gui.pack(side="top", fill="both", expand="true", padx=4, pady=4)
    gui.draw_pieces()

    #root.resizable(0,0)
    root.mainloop()

if __name__ == "__main__":
    display()
