from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.viewsets import ReadOnlyModelViewSet
from api.models.chatbot_models import ChatSession
from api.serializers.chat_session_serializers import ChatSessionSerializer

class ChatSessionView(ReadOnlyModelViewSet):
    serializer_class = ChatSessionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        print(self.kwargs['chatbot_id'])
        return ChatSession.objects.filter(chatbot__user=self.request.user, chatbot_id=self.kwargs['chatbot_id'])