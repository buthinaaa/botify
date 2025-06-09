from django.contrib import admin
from django.urls import path
from api.views.auth_views import RegisterAPIView
from rest_framework.routers import DefaultRouter
from api.views.chatbot_views import ChatbotView
from api.views.message_views import GenerateResponse, MessagesViewSet, DashboardUserMessage
from api.views.chat_session_views import ChatSessionView
router = DefaultRouter()
router.register(r'chatbots', ChatbotView, basename='chatbot')
router.register(r'chat-sessions', ChatSessionView, basename='chat_session')
router.register(r'messages', MessagesViewSet, basename='messages')

urlpatterns = [
    path('auth/register/', RegisterAPIView.as_view(), name="hello_world"), 
    path('generate-response/', GenerateResponse.as_view(), name="generate_response"),
    path('chat-sessions/<uuid:chatbot_id>/', ChatSessionView.as_view({'get': 'list'}), name="chat_session"),
    path('chat-sessions/<uuid:session_id>/messages/', MessagesViewSet.as_view({'get': 'list'}), name="messages"),
    path('chat-sessions/<uuid:session_id>/dashboard-user-message/', DashboardUserMessage.as_view(), name="dashboard_user_message"),
]

urlpatterns += router.urls
