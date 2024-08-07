# homepage/urls.py
from django.urls import path
from . import views


urlpatterns = [
    path('', views.index, name='index'),
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('recog/', views.recog_view, name='recog'),
    path('about/', views.about_view, name='about'),
    path('predict_image/', views.predict_image, name='predict_image'),
    path('predict_video/', views.predict_video, name='predict_video'),
    path('predict_youtube/', views.predict_youtube, name='predict_youtube'),
    path('webcam_feed/', views.webcam_feed, name='webcam_feed'),
]
