from django.urls import path, include
from front import views
urlpatterns = [
    path('', views.index),
    path('create/', views.index),
    path('read/1/', views.index),
]
