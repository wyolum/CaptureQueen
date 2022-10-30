import chess
import chess.pgn
import pyperclip
import webbrowser
from datetime import datetime
from datetime import date


def create_pgn(clock_board):
    initial_seconds = clock_board.initial_seconds
    increment_seconds = clock_board.increment_seconds
    game = chess.pgn.Game()
    game.headers["Event"] = "CaptureQueen Over-the-board chess capture"
    game.headers["Date"] = datetime.now().strftime("%d/%m/%y %H:%M:%S")
    game.headers["TimeControl"] = f'{initial_seconds}+{increment_seconds}'
    game.headers["Variant"] = "Standard"
    game.headers["Annotator"] = "CaptureQueen"

    outcome = clock_board.outcome()
    if outcome:
        game.headers["Termination"] = str(outcome.termination).split('.')[1]
        winner = outcome.winner
        if winner is not None:
            if winner == chess.WHITE:
                result = "1-0"
            else:
                result = "0-1"
            game.headers["Result"] = result
    if len(clock_board.move_stack) > 0:
        last_move = clock_board.move_stack[-1]
        white_ms = last_move.white_ms
        black_ms = last_move.black_ms
        if white_ms <= 0 and black_ms > 0:
            game.headers["Termination"] = "Time forfeit"
            game.headers["Result"] = "0-1"
        if black_ms <= 0 and white_ms > 0:
            game.headers["Termination"] = "Time forfeit"
            game.headers["Result"] = "1-0"
            
    node = game
    for move in clock_board.move_stack:
        node = node.add_variation(move)
    return str(game)

def upload_to_lichess(pgn):
    pyperclip.copy(pgn)
    print('paste (CTRL-V) PGN into browser window')
    webbrowser.open('https://lichess.org/paste', new=0, autoraise=True)
    
