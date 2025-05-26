from rest_framework.response import Response
from api.models.chatbot_models import Chatbot, ChatbotData
from api.serializers.message_serializer import MessageSerializer
from api.models.chatbot_models import Message
from rest_framework.views import APIView
from rest_framework import status
from drf_spectacular.utils import extend_schema, OpenApiExample
from rest_framework.exceptions import NotFound
from rest_framework.viewsets import ReadOnlyModelViewSet
from rest_framework.permissions import IsAuthenticated
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

channel_layer = get_channel_layer()

class MessagesViewSet(ReadOnlyModelViewSet):
    serializer_class = MessageSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        if self.kwargs.get('session_id'):
            return Message.objects.filter(chatbot__user=self.request.user, session_id=self.kwargs.get('session_id'))
        else:
            return Message.objects.filter(chatbot__user=self.request.user)

class GenerateResponse(APIView):
    @extend_schema(
        request=MessageSerializer,
        responses={201: MessageSerializer},
        description="Generate a bot response for a user message",
        examples=[
            OpenApiExample(
                'User message example',
                value={
                    'original_text': 'Hello, can you help me?',
                    'chatbot_id': 'uuid-string-here',
                    'session_id': 'uuid-string-here'
                },
                request_only=True,
            )
        ]
    )
    def post(self, request):
        request.data['sender'] = 'user'
        try:
            # Get the chatbot using the chatbot_id
            chatbot = Chatbot.objects.get(id=request.data['chatbot_id'])
        except Chatbot.DoesNotExist:
            raise NotFound(f"Chatbot with id {request.data['chatbot_id']} does not exist.")
        
        try:
            chatbot_data = ChatbotData.objects.get(chatbot=chatbot)
        except ChatbotData.DoesNotExist:
            raise NotFound(f"ChatbotData with id {request.data['chatbot_id']} does not exist.")
        # Get the doc_labels (intent labels from documents)
        doc_labels = chatbot_data.intent_labels
        serializer = MessageSerializer(data=request.data,context={'doc_labels': doc_labels})
        if serializer.is_valid():
            user_message = serializer.save()
            async_to_sync(channel_layer.group_send)(
                f'chat_{user_message.session_id}',
                {
                    "type": "chat_message",
                    "content": {
                        "id": str(user_message.id),
                        "session_id": str(user_message.session_id),
                        "original_text": user_message.original_text,
                        "sender": user_message.sender,
                        "timestamp": str(user_message.timestamp)
                    }
                }
            )
            original_text = user_message.original_text
            processed_text_use = user_message.processed_text_use
            ner_entities = user_message.ner_entities
            sentiment = user_message.sentiment
            overall_sentiment = user_message.overall_sentiment
            intent = user_message.intent
            #TODO: Call the chatbot model to get the response using the processed_text_use (Do all the processing you need, like NER, etc.)

            chatbot_message = {
                "original_text": "I AM BOT, HELPING YOU", #TODO: Put the chatbot model reponse here
                "sender": "bot",
                "chatbot_id": user_message.chatbot_id,
                "session_id": user_message.session_id
            }
            serializer = MessageSerializer(data=chatbot_message)
            if serializer.is_valid():
                bot_message = serializer.save()
                
                async_to_sync(channel_layer.group_send)(
                    f'chat_{user_message.session_id}',
                    {
                        "type": "chat_message",
                        "content": {
                            "id": str(bot_message.id),
                            "session_id": str(bot_message.session_id),
                            "original_text": bot_message.original_text,
                            "sender": bot_message.sender,
                            "timestamp": str(bot_message.timestamp)
                        }
                    }
                )
                
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)