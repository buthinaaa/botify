from django.contrib import admin
from django.urls import path
from api.views.auth_views import RegisterAPIView
from rest_framework.routers import DefaultRouter
from api.views.chatbot_views import ChatbotView

router = DefaultRouter()
router.register(r'chatbots', ChatbotView, basename='chatbot')


urlpatterns = [
    path('auth/register/', RegisterAPIView.as_view(), name="hello_world"), 
]

urlpatterns += router.urls
