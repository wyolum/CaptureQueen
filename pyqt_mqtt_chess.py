'''
Continue to refresh image of .board.svg
'''
import chess
import chess.svg
import time
import os.path

from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget, QApplication, QFileDialog
from PyQt5.QtWidgets import QLabel, QTextEdit, QPushButton
from PyQt5.QtCore import Qt, QTimer, QTime, QSize
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtGui import QPainter, QColor, QPen

from mqtt_clock_client import mqtt_subscribe, mqtt_start, mqtt_clock_reset
from mqtt_clock_client import MqttRenderer, mqtt_clock_pause
from mqtt_clock_client import mqtt_setblack_ms, mqtt_setwhite_ms
from mqtt_clock_client import mqtt_sethalfmove, mqtt_underpromote_to
from mqtt_clock_client import mqtt_goback, mqtt_goforward, mqtt_quit, mqtt_resign, mqtt_draw

import defaults

class PaintWidget(QWidget):
    '''
    example from https://pythonspot.com/pyqt5-colors/
    '''
    def paintEvent(self, event):
        qp = QPainter(self)
        
        qp.setPen(Qt.black)
        size = self.size()
        
        # Colored rectangles
        qp.setBrush(QColor(128, 128, 128, 50))
        qp.drawRect(0, 0, size.width(), size.height())

class MainWindow(QWidget):
    svg_filename = '.board.svg'
    def __init__(self, colors=None, flipped=False):
        super().__init__()
        if colors is None:
            colors = defaults.colors            
        self.colors = colors
        self.flipped = flipped
        self.board = chess.Board()
        self.lastmove = None
        self.svg = None
        
        self.setWindowTitle("Capture Queen")
        self.setStyleSheet(f"background-color:{self.colors['margin']};")
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setGeometry(0, 0, 650, 450)

        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(0, 0, 450, 450)

        button = QPushButton('', self)
        button.clicked.connect(self.handleRookPromotion)
        button.setIconSize(QSize(50,50))
        button.move(465,20)
        self.rook_button = button
        
        button = QPushButton('', self)
        button.clicked.connect(self.handleKnightPromotion)
        button.setIconSize(QSize(50,50))
        button.move(465,85)
        self.knight_button = button

        button = QPushButton('', self)
        button.clicked.connect(self.handleBishopPromotion)
        button.setIconSize(QSize(50,50))
        button.move(465,150) 
        self.bishop_button = button

        button = QPushButton(self)
        button.setText("Draw")
        button.move(500,250)
        button.clicked.connect(self.draw)

        button = QPushButton(self)
        button.setText("Resign White")
        button.move(500,300)
        button.clicked.connect(self.resign_white)

        button = QPushButton(self)
        button.setText("Resign Black")
        button.move(500,350)
        button.clicked.connect(self.resign_black)

        button = QPushButton(self)
        button.setText("Take back.")
        button.move(500,400)
        button.clicked.connect(mqtt_goback)

        self.grey_overlay = PaintWidget(self)
        self.grey_overlay.move(0,0)
        qsize = QSize()
        qsize.setWidth(1000)
        qsize.setHeight(1000)
        self.grey_overlay.resize(qsize)
        self.grey_overlay.setVisible(False)

        #chessboardSvg = open(self.svg_filename).read().encode('UTF-8')
        #self.widgetSvg.load(chessboardSvg)
        self.svg_stale = True
        
        self.timer = QTimer()
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.recurring_timer)
        self.timer.start()
        self.mqtt_msg_handlers = {
            'capture_queen.turn': self.mqtt_on_turn,
            'capture_queen.reset_pi': self.mqtt_on_reset_pi,
            'capture_queen.position': self.mqtt_on_position,
            
        }
        self.display_position()
        self.recurring_timer()
        mqtt_subscribe(self.on_mqtt)
        mqtt_start()
        #mqtt_clock_reset(self.initial_seconds, self.increment_seconds)
        self.hidePromotionButtons()
    def underpromote_to(self, piece):
        if self.lastmove is None:
            return
        uci = self.lastmove.uci()
        if len(uci) < 5 or uci[-1] != 'q':
            return
        fen = self.board.fen()
        if self.board.turn == chess.BLACK:
            piece = piece.upper()
            row = fen.split()[0].split('/')[0]
            row_i = 0
        else:
            piece = piece.lower()
            row_i = 7
        fields = fen.split()
        rows = fields[0].split('/')
        row = rows[row_i]
        squares = []
        for c in row:
            if c in '12345678':
                n  = int(c)
                squares += '.' * n
            else:
                squares += c
        col = ord(uci[2]) - ord('a')
        squares[col] = piece
        print(squares)
        new_row = ''
        n = 0
        for s in squares:
            if s == '.':
                n += 1
            else:
                if n > 0:
                    new_row += f'{n}'
                    n = 0
                new_row += f'{s}'
        if n > 0:
            new_row += f'{n}'
        rows[row_i] = new_row
        fields[0] = '/'.join(rows)
        new_fen =  ' '.join(fields)

        self.hidePromotionButtons()
        self.board = chess.Board(new_fen)
        self.display_position()
        mqtt_underpromote_to(piece)
        
    def handleRookPromotion(self):
        self.underpromote_to('r')

    def handleBishopPromotion(self):
        self.underpromote_to('b')

    def handleKnightPromotion(self):
        self.underpromote_to('n')
        
    def showPromotionButtons(self):
        turn = self.board.turn
        if turn == chess.BLACK:
            self.rook_button.setIcon(QIcon('./img/whiter.png'))
            self.bishop_button.setIcon(QIcon('./img/whiteb.png'))
            self.knight_button.setIcon(QIcon('./img/whiten.png'))
        else:
            self.rook_button.setIcon(QIcon('./img/blackr.png'))
            self.bishop_button.setIcon(QIcon('./img/blackb.png'))
            self.knight_button.setIcon(QIcon('./img/blackn.png'))
        self.rook_button.setVisible(True)
        self.bishop_button.setVisible(True)
        self.knight_button.setVisible(True)

    def hidePromotionButtons(self):
        self.rook_button.hide()
        self.bishop_button.hide()
        self.knight_button.hide()

    def draw(self):
        self.grey_overlay.setVisible(True)
        mqtt_draw()

    def resign_white(self):
        self.grey_overlay.setVisible(True)
        mqtt_resign(chess.WHITE)
        
    def resign_black(self):
        self.grey_overlay.setVisible(True)
        mqtt_resign(chess.BLACK)
        
    def display_position(self):
        if self.board.is_game_over():
            self.grey_overlay.setVisible(True)
        else:
            self.grey_overlay.setVisible(False)
        check = self.board.king(self.board.turn) if self.board.is_check() else None
        self.svg = chess.svg.board(self.board,size=640,
                                   check=check,
                                   flipped=self.flipped,
                                   lastmove=self.lastmove,
                                   colors=self.colors)
        
        if self.lastmove:
            lastmove_uci = self.lastmove.uci()
            if (len(lastmove_uci) == 5 and lastmove_uci[4] in 'qrbn'):
                self.showPromotionButtons()
            else:
                self.hidePromotionButtons()
            
        open(self.svg_filename, 'w').write(self.svg)
        self.svg_stale = True
        
    def mqtt_on_position(self, payload):
        uci, fen = payload.split('//')
        self.board = chess.Board(fen)
        
        if uci != 'None':
            self.lastmove = chess.Move.from_uci(uci)
        else:
            self.lastmove = None
        self.display_position()

    def mqtt_on_turn(self, payload):
        print('TURN:', payload)
    def mqtt_on_reset_pi(self, payload):
        ## set up pieces to start position
        while True:
            try:
                self.board.pop()
            except IndexError:
                break
        self.grey_overlay.setVisible(False)
        print('RESET:', payload)
    def on_mqtt(self, msg):
        '''
        store new messages at end of pending list
        '''
        print(msg.topic)
        if msg.topic in self.mqtt_msg_handlers:
            payload = msg.payload.decode('utf-8')
            self.mqtt_msg_handlers[msg.topic](payload)
        else:
            print(f'No handler for  {msg.topic}')
        
    def recurring_timer(self):
        if self.svg and self.svg_stale:
            self.widgetSvg.load(self.svg.encode('UTF-8'))
            self.svg_stale = False
        
    def keyPressEvent(self, e):
        k = e.key()
        t = e.text()
        if k == Qt.Key_Escape:
            self.close()
        elif k == Qt.Key_Right:
            mqtt_goforward()
        elif k == Qt.Key_Left:
            mqtt_goback()
        elif k == Qt.Key_Up:
            print('up')
        elif k == Qt.Key_Down:
            print('down')
        elif t == 'q':
            self.close()
        elif t == 'f':
            self.flipped = not self.flipped
            self.display_position()

    def on_right_click(self):
        print('right click')
        options = QFileDialog.Options()
        #options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,
                                                  "Open PNG Dialog",
                                                  "",
                                                  "PGN (*.pgn)",
                                                  options=options)
        if fileName:
            print(fileName)        
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            # handle the left-button press in here
            pass

        elif e.button() == Qt.MiddleButton:
            # handle the middle-button press in here.
            pass

        elif e.button() == Qt.RightButton:
            # handle the right-button press in here.
            self.on_right_click()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            pass

        elif e.button() == Qt.MiddleButton:
            pass

        elif e.button() == Qt.RightButton:
            pass

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.LeftButton:
            print("mouseDoubleClickEvent LEFT")

        elif e.button() == Qt.MiddleButton:
            print("mouseDoubleClickEvent MIDDLE")

        elif e.button() == Qt.RightButton:
            print("mouseDoubleClickEvent RIGHT")

    def close(self):
        print('closing')
        mqtt_quit()
        super().close()
if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow(flipped=False)
    window.show()
    app.exec()
