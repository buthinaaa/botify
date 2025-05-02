from django.contrib import admin
from django.urls import path
from api.views.auth_views import RegisterAPIView
from rest_framework.routers import DefaultRouter
from api.views.chatbot_views import ChatbotView
from api.views.message_views import GenerateResponse

router = DefaultRouter()
router.register(r'chatbots', ChatbotView, basename='chatbot')


urlpatterns = [
    path('auth/register/', RegisterAPIView.as_view(), name="hello_world"), 
    path('generate-response/', GenerateResponse.as_view(), name="generate_response"),
]

urlpatterns += router.urls
