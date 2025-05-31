import pickle
import numpy as np
from rest_framework.response import Response
from api.models.chatbot_models import Chatbot, ChatbotData, ChatbotDocument
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

from api.utilities.response_generation import chatbot_response

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
        doc_labels = chatbot_data.intent_labels or []

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
            previous_messages = Message.objects.filter(
                chatbot=chatbot,
                session=user_message.session
            ).only("original_text", "sender", "timestamp").order_by('-timestamp')[:10]
            conversation_context = [
                {msg.sender: msg.original_text} 
                for msg in reversed(previous_messages)
            ]
            for msg in reversed(previous_messages):
                if msg.sender == 'user':
                    conversation_context.append({"user": msg.original_text})
                elif msg.sender == 'bot':
                    conversation_context.append({"bot": msg.original_text})
            
            # Prepare document data for the chatbot_response function
            document_data = prepare_document_data(chatbot_data)
            try:
                chatbot_result = chatbot_response(
                    user_message=user_message.original_text,
                    conversation_context=conversation_context,
                    document_data=document_data,
                    business_name=chatbot.name
                )
                
                # Extract the bot response text
                bot_response_text = chatbot_result['response']['text']
                
            except Exception as e:
                # Fallback response if chatbot_response fails
                bot_response_text = "I'm sorry, I'm having trouble processing your request right now. Please try again later."
                print(f"Error in chatbot_response: {e}")

            chatbot_message = {
                "original_text": bot_response_text, #TODO: Put the chatbot model reponse here
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
        
def prepare_document_data(chatbot_data):
        """
        Prepare document data structure required by chatbot_response function
        """
        try:
            # Get all documents for this chatbot
            documents = ChatbotDocument.objects.filter(chatbot_data=chatbot_data)
            
            # Prepare raw chunks
            raw_chunks = []
            for doc in documents:
                if doc.chunks:
                    raw_chunks.extend(doc.chunks)
            
            # Deserialize indexes if they exist
            bm25_index = None
            faiss_index = None
            document_embeddings = None
            
            if chatbot_data.bm25_index_binary:
                try:
                    bm25_index = pickle.loads(chatbot_data.bm25_index_binary)
                except Exception as e:
                    print(f"Error loading BM25 index: {e}")
            
            if chatbot_data.faiss_index_binary:
                try:
                    faiss_index = pickle.loads(chatbot_data.faiss_index_binary)
                except Exception as e:
                    print(f"Error loading FAISS index: {e}")
            
            # Get document embeddings (you might need to adjust this based on your model structure)
            embeddings = []
            for doc in documents:
                if hasattr(doc, 'embedding') and doc.embedding.embedding:
                    embeddings.extend([doc.embedding.embedding])
            
            if embeddings:
                document_embeddings = np.array(embeddings)
            
            # Prepare tokenized docs (if needed)
            tokenized_docs = []
            for doc in documents:
                if doc.tokenized_text:
                    tokenized_docs.extend(doc.tokenized_text)
            
            return {
                'raw_chunks': raw_chunks,
                'bm25_index': bm25_index,
                'tokenized_docs': tokenized_docs,
                'faiss_index': faiss_index,
                'document_embeddings': document_embeddings,
                'intent_labels': chatbot_data.intent_labels or []
            }
            
        except Exception as e:
            print(f"Error preparing document data: {e}")
            return {
                'raw_chunks': [],
                'bm25_index': None,
                'tokenized_docs': None,
                'faiss_index': None,
                'document_embeddings': None,
                'intent_labels': []
            }