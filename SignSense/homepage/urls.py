# homepage/urls.py
from django.urls import path
from . import views
from .views import predict

urlpatterns = [
    path('', views.index, name='index'),
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('recog/', views.recog_view, name='recog'),
    path('about/', views.about_view, name='about'),
    path('api/predict/', predict, name='predict'),
]
