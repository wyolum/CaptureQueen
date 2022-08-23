import os

import cairosvg
import pygame
import io
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

class PygameRender:
    def __init__(self, size):
        pygame.init()
        self.size = size
        self.window = pygame.display.set_mode((size, size), pygame.NOFRAME)
        self.screen = pygame.display.get_surface()

    def render(self, board, flipped, colors=None):
        if len(board.move_stack) > 0:
            lastmove = board.move_stack[-1]
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
