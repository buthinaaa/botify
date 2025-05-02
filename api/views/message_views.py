import uuid
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.viewsets import ModelViewSet
from api.models.chatbot_models import ChatSession, Chatbot, ChatbotDocument
from api.utilities.message_processing import preprocess_message
from api.models.chatbot_models import Message
from api.serializers.message_serializer import MessageSerializer
from django.utils import timezone
from datetime import timedelta
from rest_framework.permissions import AllowAny
from rest_framework.exceptions import ValidationError
from rest_framework.views import APIView
from rest_framework import status
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiExample
from drf_spectacular.types import OpenApiTypes

class GenerateResponse(APIView):
    @extend_schema(
        request=MessageSerializer,
        responses={201: MessageSerializer},
        description="Generate a bot response for a user message",
        examples=[
            OpenApiExample(
                'User message example',
                value={
                    'original_text': 'Hello, how can you help me?',
                    'chatbot_id': 'uuid-string-here',
                    'session_id': 'uuid-string-here'
                },
                request_only=True,
            )
        ]
    )
    def post(self, request):
        request.data['sender'] = 'user'
        serializer = MessageSerializer(data=request.data)
        if serializer.is_valid():
            user_message = serializer.save()
            original_text = user_message.original_text
            processed_text_use = user_message.processed_text_use
            #TODO: Call the chatbot model to get the response using the processed_text_use (Do all the processing you need, like NER, etc.)

            chatbot_message = {
                "original_text": "I AM BOT, HELPING YOU", #TODO: Put the chatbot model reponse here
                "sender": "bot",
                "chatbot_id": user_message.chatbot_id,
                "session_id": user_message.session_id
            }
            serializer = MessageSerializer(data=chatbot_message)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)