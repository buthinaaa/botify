from rest_framework import serializers
from api.models.chatbot_models import Message, ChatSession, Chatbot
from api.utilities.message_processing import preprocess_message

class MessageSerializer(serializers.ModelSerializer):
    chatbot_id = serializers.UUIDField(write_only=True)
    original_text = serializers.CharField()
    session_id = serializers.UUIDField(allow_null=True)

    class Meta:
        model = Message
        fields = ['id', 'session_id', 'original_text', 'sender', 'timestamp', 'chatbot_id']

    def create(self, validated_data):
        chatbot_id = validated_data.pop('chatbot_id')
        session_id = validated_data.pop('session_id', None)
        chatbot = Chatbot.objects.get(id=chatbot_id)
        if session_id:
            session = ChatSession.objects.get(id=session_id)
        else:
            session = ChatSession.objects.create(chatbot=chatbot)
        
        message = Message(session=session, chatbot=chatbot, **validated_data)
        if validated_data.get('sender') == 'user':
            preprocessed_data = preprocess_message(validated_data.get('original_text'))
            message.processed_text_use = preprocessed_data['text_use']
        message.save()
        return message
