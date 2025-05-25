from rest_framework import serializers
from api.models.chatbot_models import Message, ChatSession, Chatbot
from api.utilities.intent_recognition import detect_intent
from api.utilities.message_processing import preprocess_message
from api.utilities.ner import extract_ner_entities
from api.utilities.sentiment_analysis import sentiment_pipeline

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
            message.processed_text_use = preprocessed_data.get('text_use', '')

            # Build context from previous user messages
            previous_messages = Message.objects.filter(session=session, sender="user").exclude(original_text=validated_data.get('original_text')).order_by('-timestamp')[:5]
            context = [msg.processed_text_use for msg in previous_messages if msg.processed_text_use]

            ner_entities = extract_ner_entities({
                "original": message.original_text,
                "text_use": message.processed_text_use
            }, context=context)
            message.ner_entities = ner_entities

            sentiment_result = sentiment_pipeline(message.processed_text_use, context=context or [])
            message.sentiment = sentiment_result.get('sentiment', '')
            message.overall_sentiment = sentiment_result.get('overall_sentiment', {})

            doc_labels = self.context.get('doc_labels', [])
            intent_result = detect_intent(message.processed_text_use, doc_labels)
            matched_labels = intent_result.get('matched_labels', [])
            if isinstance(matched_labels, str):
                matched_labels = [matched_labels]
            elif matched_labels is None:
                matched_labels = []
            message.intent = matched_labels

        message.save()
        return message
