'''
Continue to refresh image of .board.svg
'''
import time
import os.path

from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QLabel

from defaults import colors


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Capture Queen")
        self.setStyleSheet(f"background-color:{colors['margin']};")
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setGeometry(0, 0, 600, 450)

        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(0, 0, 450, 450)

        p1_label = QLabel('Player 1', self)
        p1_label.move(460, 0)
        p2_label = QLabel('Player 2', self)
        p2_label.move(460, 430)

        chessboardSvg = open('.board.svg').read().encode('UTF-8')
        self.widgetSvg.load(chessboardSvg)

        self.last_ctime = 0
        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.recurring_timer)
        self.timer.start()
        

    def recurring_timer(self):
        if os.path.exists('.board.svg'):
            ctime = os.path.getctime(".board.svg")
            if ctime > self.last_ctime:
                self.last_ctime = ctime
                svg = open('.board.svg').read().encode('UTF-8')
                self.widgetSvg.load(svg)
        
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()
            
if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
