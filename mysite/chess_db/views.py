from django.http import HttpResponse
from django.shortcuts import render
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

from .models import Game, Move
# Create your views here.
import chess_db_functions

def index(request):
    page = request.GET.get('page', 1)
    latest_game_list = Game.objects.order_by('-Date')
    keepers = []
    for i, game in enumerate(latest_game_list):
        n_move = Move.objects.filter(game_id=game.id).count()
        if n_move > 0:
            keepers.append(i)
    latest_game_list = [latest_game_list[i] for i in keepers]
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
    move = Move.objects.filter(game_id=game_id).filter(ply=ply)
    svg = chess_db_functions.get_image_at(game_id, ply, size=350)
    if ply >= 0:
        prev_ply = ply - 1
    else:
        prev_ply = -1
    if ply < game.max_ply:
        next_ply = ply + 1
    else:
        next_ply = game.max_ply
    context = {'game':game,
               'ply':ply,
               'prev_ply':prev_ply,
               'next_ply':next_ply,
               'svg':svg}
    return render(request, 'chess_db/move.html', context)

#   response = "You're looking at the results of question %s."
#   return HttpResponse(response % game_id)
