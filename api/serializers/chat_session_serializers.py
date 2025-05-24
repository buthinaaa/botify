from rest_framework import serializers
from api.models.chatbot_models import ChatSession
from api.serializers.chatbot_serializers import ChatbotSerializer

class ChatSessionSerializer(serializers.ModelSerializer):
    chatbot = ChatbotSerializer(read_only=True)
    class Meta:
        model = ChatSession
        fields = ['id', 'chatbot', 'created_at']
