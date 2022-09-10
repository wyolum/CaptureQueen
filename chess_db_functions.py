import io
import chess.svg
import chess.pgn
from datetime import datetime
import sqlite3
from collections import defaultdict
import defaults

DB_FILENAME = 'chess.db'
BASE_DIR = 'home/pi/code/mysite'
DB_FILENAME = '/home/pi/code/CaptureQueen/mysite/db.sqlite3'
def rename(name):
    return f'chess_db_{name.lower()}'

def get_db():
    db = sqlite3.connect(DB_FILENAME)
    return db
get_db()

def get_game_id(date):
    db = get_db()
    sql = f'''\
    INSERT INTO {rename("Game")}
        (Event, Site, Date, White, Black, 
         Result, Termination, Annotator, Variant, TimeControl) 
    VALUES 
        ("CaptureQueen", "NA", "{date}", "White", "Black",
         "NA", "NA", "NA", "Standard", "300+0")'''
    db.execute(sql)
    sql = f'SELECT max(rowid) FROM {rename("Game")}'
    out = db.execute(sql).fetchone()[0]
    db.commit()
    return out

def update_game(game_id, pgn_dict):
    db = get_db()
    sql = f'''\
    UPDATE {rename("Game")}
    SET 
        white="{pgn_dict["White"]}",
        black="{pgn_dict["Black"]}",
        site="{pgn_dict["Site"]}",
        date="{pgn_dict["Date"]}",
        result="{pgn_dict["Result"]}",
        termination="{pgn_dict["Termination"]}",
        timecontrol="{pgn_dict["TimeControl"]}"
    WHERE
        rowid={game_id}
    '''
    db.execute(sql)
    db.commit()

def move(game_id, ply, clockmove):
    db = get_db()
    sql = f'''\
        DELETE FROM {rename("Move")}
        WHERE game_id={game_id} AND ply >= {ply}
        '''
    db.execute(sql)
    sql = f'''\
        INSERT INTO {rename("Move")}
            (game_id, ply, clockmove)
        VALUES
            ({game_id}, {ply}, "{clockmove}")
        '''
    db.execute(sql)
    db.commit()

def get_pgn(game_id):
    if game_id is None:
        return "1."
    db = get_db()
    sql = f'SELECT * FROM {rename("Game")} WHERE rowid = {game_id}'
    cur = db.execute(sql)
    row = cur.fetchone()
    cols = [r[0] for r in cur.description]
    out = []
    for i, col in enumerate(cols):
        if col != 'id':
            out.append(f'[{col} "{row[i]}"]')
    out.append(f'{{game_id "{game_id}"}}')
    sql = f'''\
    SELECT 
       ply, clockmove
    FROM 
        {rename("Move")} 
    WHERE
        game_id={game_id}
    ORDER BY
        ply
    '''
    cur = db.execute(sql)
    for ply, clockmove in cur.fetchall():
        if ply % 2 == 0: ## white, new row
            if 'resigns' in clockmove:
                pass
            else:
                out.append(f'{ply//2 + 1}. ')
        out[-1] = out[-1] + f'{clockmove} '
    return '\n'.join(out)

def list_games(min_moves=0):
    db = get_db()
    sql = f'SELECT game_id, max(ply) as n_move FROM {rename("Move")} GROUP BY game_id'
    cur = db.execute(sql)
    result = [row for row in cur.fetchall() if row[1] >= min_moves]
    out = []
    for game_id, n_move in result:
        sql = f'SELECT rowid, White, Black, Result, Termination, TimeControl from {rename("Game")} WHERE rowid={game_id}'
        cur = db.execute(sql)
        row = list(cur.fetchone())
        row.insert(1, n_move)
        out.append(row)
    return out

def get_game(game_id):
    if game_id is None:
        game = chess.pgn.read_game(io.StringIO('1. '))
        board = game.board()
    else:
        pgn = get_pgn(game_id)
        game = chess.pgn.read_game(io.StringIO(pgn))
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
    return game, board

def get_final_image(game_id, size=640, flipped=False):
    game, board = get_game(game_id)
    if len(board.move_stack) > 0:
        lastmove = board.move_stack[-1]
    else:
        lastmove = None
    svg = chess.svg.board(board,size=size,
                          flipped=flipped,
                          lastmove=lastmove,
                          colors=defaults.colors)
    return svg

def get_image_at(game_id, ply=-1, size=640, flipped=False):
    game, board = get_game(game_id)
    out = chess.Board()
    for i in range(min([ply+1, len(board.move_stack)])):
        out.push(board.move_stack[i])
    if len(out.move_stack) > 0:
        lastmove = out.move_stack[-1]
    else:
        lastmove = None
    svg = chess.svg.board(out,size=size,
                          flipped=flipped,
                          lastmove=lastmove,
                          colors=defaults.colors)
    return svg

def test():
    date = datetime.now().strftime("%d/%m/%y %H:%M:%S")
    game_id = get_game_id(date)
    pgn_dict = defaultdict(lambda : "?")
    pgn_dict['Black'] = '{black}'
    pgn_dict['White'] = 'wyojustin'
    pgn_dict['Date'] = date
    pgn_dict['TimeControl'] = '300+0'
    move(game_id, 0, 'e4 {297.3  }')
    move(game_id, 1, 'f5 {297.3  }')
    move(game_id, 2, 'd4 {297.3  }')
    move(game_id, 3, 'g5 {297.3  }')
    move(game_id, 4, 'Qh5# {297.3  }')
    pgn_dict["termination"] = "CHECKMATE"
    pgn_dict["result"] = "1-0"
    update_game(game_id, pgn_dict)
    print(get_pgn(game_id))
    game_id = 315
    game, board = get_game(game_id)
    print(game)
    print(board)
    open('.junk.svg', 'w').write(get_final_image(game_id, 250, flipped=True))
if __name__ == '__main__':
    test()
