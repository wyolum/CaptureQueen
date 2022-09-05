from datetime import datetime
import sqlite3
from collections import defaultdict

DB_FILENAME = 'chess.db'

def get_db():
    db = sqlite3.connect(DB_FILENAME)
    return db

def create_table(name, cols):
    db = get_db()
    columns = ','.join([' '.join(row) for row in cols])
    sql = f'CREATE TABLE IF NOT EXISTS {name} ({columns})'
    db.execute(sql)
    
def create_db():
    db = get_db()
    game_cols = [
        ['Event', 'VARCHAR'],
        ['Site', 'VARCHAR'],
        ['Date', 'DATETIME'],
        ['White', 'VARCHAR'],
        ['Black', 'VARCHAR'],
        ['Result', 'VARCHAR'],
        ['Termination', 'VARCHAR'],
        ['TimeControl', 'VARCHAR'],
        ['Annotator', 'VARCHAR'],
        ['Variant', 'VARCHAR'],
    ]
    create_table('Game', game_cols)
    
    move_cols = [
        ['gameid', 'INT', 'NOT NULL'],
        ['ply', 'INT', 'NOT NULL'],
        ['clockmove', 'VARCHAR(40)', 'NOT NULL'],
        ['FOREIGN KEY(gameid) REFERENCES Game(rowid)']
        ]
    create_table('Move', move_cols)
    sql = f'''\
        CREATE UNIQUE INDEX 
        IF NOT EXIST 
            gameid_ply 
        ON 
            Move(gameid, ply)'''
create_db()

def get_gameid(date):
    db = get_db()
    sql = f'''\
    INSERT INTO Game 
        (date) 
    VALUES 
        ("{date}")'''
    db.execute(sql)
    sql = 'SELECT max(rowid) FROM Game'
    out = db.execute(sql).fetchone()[0]
    db.commit()
    return out

def update_game(gameid, pgn_dict):
    db = get_db()
    sql = f'''\
    UPDATE Game
    SET 
        white="{pgn_dict["White"]}",
        black="{pgn_dict["Black"]}",
        site="{pgn_dict["Site"]}",
        date="{pgn_dict["Date"]}",
        result="{pgn_dict["Result"]}",
        termination="{pgn_dict["Termination"]}",
        timecontrol="{pgn_dict["TimeControl"]}"
    WHERE
        rowid={gameid}
    '''
    db.execute(sql)
    db.commit()

def move(gameid, ply, clockmove):
    db = get_db()
    sql = f'''\
        DELETE FROM Move
        WHERE gameid={gameid} AND ply >= {ply}
        '''
    db.execute(sql)
    sql = f'''\
        INSERT INTO Move
            (gameid, ply, clockmove)
        VALUES
            ({gameid}, {ply}, "{clockmove}")
        '''
    db.execute(sql)
    db.commit()

def get_pgn(gameid):
    db = get_db()
    sql = f'SELECT * FROM Game WHERE rowid = {gameid}'
    cur = db.execute(sql)
    row = cur.fetchone()
    cols = [r[0] for r in cur.description]
    out = []
    for i, col in enumerate(cols):
        out.append(f'[{col} "{row[i]}"]')
    out.append(f'[gameid "{gameid}"]')
    sql = f'''\
    SELECT 
       ply, clockmove
    FROM 
        Move 
    WHERE
        gameid={gameid}
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

def test():
    date = datetime.now().strftime("%d/%m/%y %H:%M:%S")
    gameid = get_gameid(date)
    pgn_dict = defaultdict(lambda : "?")
    pgn_dict['black'] = '{black}'
    pgn_dict['white'] = 'wyojustin'
    pgn_dict['date'] = date
    pgn_dict['timecontrol'] = '300+0'
    move(gameid, 0, 'e4 {297.3  }')
    move(gameid, 1, 'f5 {297.3  }')
    move(gameid, 2, 'd4 {297.3  }')
    move(gameid, 3, 'g5 {297.3  }')
    move(gameid, 4, 'Qh5# {297.3  }')
    pgn_dict["termination"] = "CHECKMATE"
    pgn_dict["result"] = "1-0"
    update_game(gameid, pgn_dict)
    print(get_pgn(gameid))

if __name__ == '__main__':
    test()
