from django.http import HttpResponse
from django.shortcuts import render
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
import sqlite3

from .models import Game, Move
# Create your views here.
import chess_db_functions

def index(request):
    page = request.GET.get('page', 1)
    if False:
        sql = ''''\
    SELECT game_id, max(ply) as max_ply
    FROM Move
    WHERE max_ply > 0
    GROUP BY game_id
    '''
        db = sqlite3.connect('/home/pi/code/CaptureQueen/mysite/db.sqlite3')
        cur = db.execute(sql)
        result = cur.fetchall()

    latest_game_list = Game.objects.filter(MoveCount__gte=3).order_by('-Date')
    paginator = Paginator(latest_game_list, 6)
    games = paginator.page(page)
    
    #context = {'latest_question_list': latest_question_list}
    context = {'games': games}
    #svg = chess_db_functions.get_final_image(1, size=250)
    #return HttpResponse(svg)
    return render(request, 'chess_db/index.html', context)

def game(request, game_id):
    game = Game.objects.get(pk=game_id)
    context = {'game': game}
    return render(request, 'chess_db/game.html', context)

def newgame(request, game_id, ply=None):
    return move(request, game_id, ply=-1)

def move(request, game_id, ply=None):
    game = Game.objects.get(pk=game_id)
    if ply is None:
        ply = game.max_ply
    moveset = Move.objects.filter(game_id=game_id).filter(ply=ply)
    if moveset.count() > 0:
        move= moveset[0]
    else:
        move='1.'
    svg = chess_db_functions.get_image_at(game_id, ply, size=250)
    if ply >= 0:
        prev_ply = ply - 1
    else:
        prev_ply = -1
    if ply < game.max_ply:
        next_ply = ply + 1
    else:
        next_ply = game.max_ply
    context = {'game':game,
               'move':str(move),
               'ply':ply,
               'prev_ply':prev_ply,
               'next_ply':next_ply,
               'svg':svg}
    return render(request, 'chess_db/move.html', context)

#   response = "You're looking at the results of question %s."
#   return HttpResponse(response % game_id)
