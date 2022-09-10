from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('/<int:game_id>/', views.move, name='detail'),
    path('/<int:game_id>/<int:ply>/', views.move, name='move'),
    path('/<int:game_id>/-1/', views.newgame, name='move'),
]
