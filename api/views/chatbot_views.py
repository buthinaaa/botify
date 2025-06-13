from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.viewsets import ModelViewSet
from api.models.chatbot_models import Chatbot
from api.serializers.chatbot_serializers import ChatbotSerializer

class ChatbotView(ModelViewSet):
    queryset = Chatbot.objects.all()
    serializer_class = ChatbotSerializer
    parser_classes = (MultiPartParser, FormParser)

    def get_queryset(self):
        queryset = super().get_queryset()
        return queryset.filter(user=self.request.user) if self.action != 'retrieve' else queryset

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context['request'] = self.request
        return context

    def get_permissions(self):
        if self.action == 'retrieve':
            return [AllowAny()]
        return [IsAuthenticated()]