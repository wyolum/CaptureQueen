import os

import cairosvg
import pygame
import io
import chess
import chess.svg
pos_x = 1900
pos_y = 100
os.environ['SDL_VIDEO_WINDOW_POS'] = '%i,%i' % (pos_x,pos_y)

DPI = 96

def render_svg(_svg, _scale):
    _svg = cairosvg.svg2svg(_svg, dpi = (DPI / _scale))
    _bytes = cairosvg.svg2png(_svg)
    byte_io = io.BytesIO(_bytes)
    out = pygame.image.load(byte_io)
    return out

def uci2ints(uci, flipped):
    square = chess.parse_square(uci)
    #square = chess.square(num)
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    col = 7 - (file)
    row = rank
    if not flipped:
        row = 7 - row
        col = 7 - col
    return col, row

class PygameRender:
    def __init__(self, size):
        pygame.init()
        self.size = size
        self.window = pygame.display.set_mode((size, size), pygame.NOFRAME)
        self.screen = pygame.display.get_surface()
        self.offset = size / 26.38888888888889
        self.side= (size - 2 * self.offset) / 8.
        self.from_square = pygame.Rect(self.offset, self.offset,
                                       self.side, self.side)
        self.to_square = pygame.Rect(self.offset, self.offset,
                                       self.side, self.side)
        
    def render(self, board, flipped, colors=None):
        if board is None:
            display_ready = True
            board = chess.Board()
        else:
            display_ready = False
        if len(board.move_stack) > 0:
            lastmove = board.move_stack[-1]
            uci = lastmove.uci()
            frm = uci2ints(uci[0:2], flipped)
            _to = uci2ints(uci[2:4], flipped)
            from_square = self.from_square.move(frm[0] * self.side,
                                                frm[1] * self.side)
            to_square = self.from_square.move(_to[0] * self.side,
                                              _to[1] * self.side)
            width = 3
        
            pygame.draw.rect(self.screen, (255, 128, 128), from_square,
                             width)
            pygame.draw.rect(self.screen, (255, 128, 128), to_square,
                             width)
            pygame.display.flip()
            print('flipped')
        else:
            lastmove = None
        if colors is None:
            colors = {'square dark':"#aaaaaa",
                      'square light':"#ffffff"}

        svg = chess.svg.board(board,size=self.size, flipped=flipped,
                              lastmove=lastmove, colors=colors)
        #open("junk.svg", 'w').write(svg);print("wrote junk.svg");return
        image = render_svg(svg, 1)
        self.screen.blit(image, (0, 0))

        if display_ready:
            white = (255, 255, 255)
            green = (0, 255, 0)
            blue = (0, 0, 128)
            font = pygame.font.Font('freesansbold.ttf', 32)
            text = font.render('Ready...', True, green, blue)
            textRect = text.get_rect()
            textRect.center = (self.size // 2, self.size // 2)
            self.screen.blit(text, textRect)
            
        pygame.display.flip()
        pygame.display.update()
        
    def exit(self):
        raise SystemExit

if __name__ == '__main__':
    import time
    pgr = PygameRender(500)
    import chess
    board = chess.Board()
    board.push_uci("e2e4")
    while 1:
        pgr.render(board, True)
        time.sleep(3)
        print('tick')
        pgr.render(board, False)
        time.sleep(3)
        print('tick')
