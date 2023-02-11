from django.db import models

import os.path
import sys
mydir = os.path.dirname(__file__)
capture_queen_dir = os.path.abspath(os.path.join(mydir, '../../'))
sys.path.append(capture_queen_dir)
import chess_db_functions

# Create your models here.

class Game(models.Model):
    Event = models.CharField(max_length=80)
    Site = models.CharField(max_length=80)
    Date = models.DateTimeField()
    White = models.CharField(max_length=80)
    Black = models.CharField(max_length=80)
    Result = models.CharField(max_length=20)
    Termination = models.CharField(max_length=80)
    Annotator = models.CharField(max_length=80)
    Variant = models.CharField(max_length=80, default="Standard")
    TimeControl = models.CharField(max_length=40, default="300+0")
    MoveCount = models.IntegerField(default=0)
    
    def __init__(self, *args, **kw):
        models.Model.__init__(self, *args,**kw)
        self.n_move = Move.objects.filter(game_id=self.id).count()
        self.max_ply = self.n_move - 1
        self.pgn = chess_db_functions.get_pgn(self.id)
        lines = self.pgn.splitlines()
        if len(lines) > 14:
            lines = lines[:12] + ['...', lines[-1]]
        self.abreviated_pgn = '\n'.join(lines)
        self.final_image = chess_db_functions.get_final_image(self.id, size=250)
    def __str__(self):
        return f'''\
Game("{self.id},"{self.White}","{self.Black}", ...)'''

    def format_header(self):
        return f'''\
[Event "{self.Event}"]
[Site "{self.Site}"]
[Date "{self.Date}"]
[White "{self.White}"]
[Black "{self.Black}"]
[Result "{self.Result}"]
[Termination "{self.Termination}"]
[Annotator "{self.Annotator}"]
[Variant "{self.Variant}"]
[id "{self.id}"]\
'''
        
    def get_pgn(self):
        moves = self.move_set.all().order_by('ply')
        out = [self.format_header()]
        
        for move in moves:
            out.append(str(move))
        return '\n'.join(out)


class Move(models.Model):
    game = models.ForeignKey(Game, on_delete=models.CASCADE)
    ply = models.IntegerField()
    clockmove = models.CharField(max_length=100)

    class Meta:
        unique_together = ('game_id', 'ply')
        
    def __init__(self, *args, **kw):
        models.Model.__init__(self, *args,**kw)
        
    def __str__(self):
        move_number = self.ply // 2 + 1
        if self.ply % 2 == 0:
            out = f'{move_number}. {self.clockmove}'
        else:
            out = f'{move_number}. ... {self.clockmove}'
        return out
