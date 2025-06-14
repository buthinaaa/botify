import pickle
import re
import tempfile
import numpy as np
import logging
import time
import traceback
from rest_framework.response import Response
from api.models.chatbot_models import ChatSession, Chatbot, ChatbotData, ChatbotDocument
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
import faiss
from django.db import transaction
import os
from django.core.mail import send_mail
from django.conf import settings
from django.template.loader import render_to_string
from django.utils.html import strip_tags

from api.utilities.response_generation import chatbot_response
from api.utilities.sentiment_analysis import check_for_fallback

# Configure logger
logger = logging.getLogger(__name__)

channel_layer = get_channel_layer()

class MessagesViewSet(ReadOnlyModelViewSet):
    serializer_class = MessageSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        start_time = time.time()
        logger.warning(f"MessagesViewSet.get_queryset called by user: {self.request.user.id}")
        
        try:
            session_id = self.kwargs.get('session_id')
            user_id = self.request.user.id
            
            if session_id:
                logger.warning(f"Filtering messages for user {user_id} and session {session_id}")
                queryset = Message.objects.filter(chatbot__user=self.request.user, session_id=session_id)
                for message in queryset:
                    print(message.original_text)
                message_count = queryset.count()
                logger.warning(f"Found {message_count} messages for user {user_id} and session {session_id}")
            else:
                logger.warning(f"Filtering all messages for user {user_id}")
                queryset = Message.objects.filter(chatbot__user=self.request.user)
                message_count = queryset.count()
                logger.warning(f"Found {message_count} total messages for user {user_id}")
            
            end_time = time.time()
            logger.warning(f"MessagesViewSet.get_queryset completed in {end_time - start_time:.3f}s")
            return queryset
            
        except Exception as e:
            logger.error(f"Error in MessagesViewSet.get_queryset for user {self.request.user.id}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

class DashboardUserMessage(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request, session_id):
        message = request.data.get('message')
        chat_session = ChatSession.objects.get(id=session_id)
        dashboard_user_message = {
            "original_text": message,
            "sender": "admin",
            "chatbot_id": chat_session.chatbot_id,
            "session_id": session_id
            }
        serializer = MessageSerializer(data=dashboard_user_message)
        serializer.is_valid(raise_exception=True)

        with transaction.atomic():
            serializer.save()
            chat_session.is_intervened = True
            chat_session.save()
        async_to_sync(channel_layer.group_send)(
            f'chat_{session_id}',
            {
                "type": "chat_message",
                "content": {
                    "id": str(serializer.data['id']),
                    "session_id": str(session_id),
                    "original_text": serializer.data['original_text'],
                    "sender": serializer.data['sender'],
                    "timestamp": str(serializer.data['timestamp'])
                }
            }
        )
        return Response(status=status.HTTP_201_CREATED)


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
        start_time = time.time()
        request_id = id(request)  # Unique identifier for this request
        
        try:
            request.data['sender'] = 'user'
            chatbot_id = request.data.get('chatbot_id')
            session_id = request.data.get('session_id')
            original_text = request.data.get('original_text', '')
            
            try:
                chatbot = Chatbot.objects.get(id=chatbot_id)
            except Chatbot.DoesNotExist:
                raise NotFound(f"Chatbot with id {chatbot_id} does not exist.")
            try:
                chatbot_data = ChatbotData.objects.get(chatbot=chatbot)
                logger.warning(f"[Request {request_id}] Found ChatbotData with {len(chatbot_data.intent_labels or [])} intent labels")
            except ChatbotData.DoesNotExist:
                logger.error(f"[Request {request_id}] ChatbotData with id {chatbot_id} does not exist")
                raise NotFound(f"ChatbotData with id {chatbot_id} does not exist.")
            
            doc_labels = chatbot_data.intent_labels or []
            serializer = MessageSerializer(data=request.data, context={'doc_labels': doc_labels})

            if serializer.is_valid():
                user_message = serializer.save()
                
                try:
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
                except Exception as ws_error:
                    logger.error(f"[Request {request_id}] WebSocket send failed: {str(ws_error)}")

                previous_messages = Message.objects.filter(
                    chatbot=chatbot,
                    session=user_message.session
                ).values('sender', 'sentiment', 'original_text').order_by('-timestamp')[:10]
                previous_messages = [
                    {
                        'sentiment': msg['sentiment'],
                        msg['sender']: msg['original_text']
                    }
                    for msg in previous_messages
                ]
                if check_for_fallback(previous_messages):
                    try:
                        session_id = user_message.session_id
                        conversation_url = f"{settings.FRONTEND_BASE_URL}{settings.FRONTEND_CHATS_PATH}/{chatbot.id}?session_id={session_id}"                        
                        html_message = render_to_string('emails/human_intervention_alert.html', {
                            'chatbot_name': chatbot.name,
                            'user_email': chatbot.user.email,
                            'conversation_url': conversation_url,
                        })
                        plain_message = strip_tags(html_message)
                        
                        send_mail(
                            subject="🚨 Botify Alert: Human Intervention Needed",
                            message=plain_message,
                            from_email=settings.DEFAULT_FROM_EMAIL,
                            recipient_list=[chatbot.user.email],
                            html_message=html_message,
                            fail_silently=False,
                        )
                        logger.warning(f"[Request {request_id}] Human intervention alert email sent to {chatbot.user.email}")
                    except Exception as email_error:
                        logger.error(f"[Request {request_id}] Failed to send human intervention email: {str(email_error)}")
                document_data_start = time.time()
                document_data = prepare_document_data(chatbot_data, request_id)
                document_data_end = time.time()
                chatbot_start_time = time.time()
                chatbot_session = ChatSession.objects.filter(id=session_id).first() if session_id else None
                logger.warning(f"[Request {request_id}] PRINTING DOCPRINTING DOCPRINTING DOCPRINTING DOCPRINTING DOCPRINTING DOCPRINTING DOCPRINTING DOCPRINTING DOCPRINTING DOCPRINTING DOCPRINTING DOCPRINTING DOCPRINTING DOCPRINTING DOCPRINTING DOCPRINTING DOCPRINTING DOCPRINTING DOCPRINTING DOCPRINTING DOCPRINTING DOCDocument Data: {document_data}")
                if not chatbot_session or not chatbot_session.is_intervened:
                    try:
                        chatbot_result = chatbot_response(
                            user_message=user_message.original_text,
                            conversation_context=previous_messages,
                            document_data=document_data,
                            business_name=chatbot.name
                        )
                        chatbot_end_time = time.time()
                        logger.warning(f"[Request {request_id}] Chatbot response generated in {chatbot_end_time - chatbot_start_time:.3f}s")
                        
                        # Extract the bot response text
                        bot_response_text = chatbot_result['response']['text']
                        bot_response_text = re.split(
                            r'\buser\w*\b(?:\s*\([^)]*\))?(?:\s*[:]|(?:\s+\w+))',
                            bot_response_text,
                            flags=re.IGNORECASE
                        )[0]
                        bot_response_text = re.sub(r'\s*\([^)]*\)', '', bot_response_text)
                        logger.warning(f"[Request {request_id}] Bot response: '{bot_response_text[:100]}...'")
                        logger.warning(f"[Request {request_id}] Response debug info: {chatbot_result.get('debug_info', {})}")
                        
                    except Exception as e:
                        chatbot_end_time = time.time()
                        logger.error(f"[Request {request_id}] Chatbot response failed after {chatbot_end_time - chatbot_start_time:.3f}s: {str(e)}")
                        logger.error(f"[Request {request_id}] Chatbot error traceback: {traceback.format_exc()}")
                        
                        # Fallback response if chatbot_response fails
                        bot_response_text = "I'm sorry, I'm having trouble processing your request right now. Please try again later."

                    # Create and save bot response
                    logger.warning(f"[Request {request_id}] Creating bot response message")
                    chatbot_message = {
                        "original_text": bot_response_text,
                        "sender": "bot",
                        "chatbot_id": user_message.chatbot_id,
                        "session_id": user_message.session_id
                    }

                    serializer = MessageSerializer(data=chatbot_message)

                    if serializer.is_valid():
                        logger.warning(f"[Request {request_id}] Bot message serialization valid, saving to database")
                        bot_message = serializer.save()
                        logger.warning(f"[Request {request_id}] Bot message saved with ID: {bot_message.id}")
                        
                        # Send bot message to WebSocket
                        logger.warning(f"[Request {request_id}] Sending bot message to WebSocket")
                        try:
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
                            logger.warning(f"[Request {request_id}] Bot message sent to WebSocket successfully")
                        except Exception as ws_error:
                            logger.error(f"[Request {request_id}] WebSocket send failed for bot message: {str(ws_error)}")
                        
                        end_time = time.time()
                        total_time = end_time - start_time
                        logger.warning(f"[Request {request_id}] GenerateResponse.post completed successfully in {total_time:.3f}s")
                        
                        return Response(serializer.data, status=status.HTTP_201_CREATED)
                    else:
                        logger.error(f"[Request {request_id}] Bot message serialization failed: {serializer.errors}")
                else:
                    logger.warning(f"[Request {request_id}] Chatbot session is intervened, skipping response generation")
                    return Response({"original_text": None}, status=status.HTTP_200_OK)
            else:
                logger.error(f"[Request {request_id}] User message serialization failed: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
                
        except Exception as e:
            end_time = time.time()
            total_time = end_time - start_time
            logger.error(f"[Request {request_id}] GenerateResponse.post failed after {total_time:.3f}s: {str(e)}")
            logger.error(f"[Request {request_id}] Full traceback: {traceback.format_exc()}")
            raise
        
def prepare_document_data(chatbot_data, request_id=None):
    """
    Prepare document data structure required by chatbot_response function
    """
    logger.warning(f"[Request {request_id}] Starting prepare_document_data")
    start_time = time.time()
    
    try:
        # Get all documents for this chatbot
        logger.warning(f"[Request {request_id}] Fetching ChatbotDocument objects")
        documents = ChatbotDocument.objects.filter(chatbot_data=chatbot_data)
        doc_count = documents.count()
        logger.warning(f"[Request {request_id}] Found {doc_count} documents")
        
        # Prepare raw chunks
        logger.warning(f"[Request {request_id}] Preparing raw chunks")
        raw_chunks = []
        for i, doc in enumerate(documents):
            if doc.chunks:
                chunk_count = len(doc.chunks)
                raw_chunks.extend(doc.chunks)
                logger.warning(f"[Request {request_id}] Document {i+1}: Added {chunk_count} chunks")
            else:
                logger.warning(f"[Request {request_id}] Document {i+1}: No chunks found")
        
        total_chunks = len(raw_chunks)
        logger.warning(f"[Request {request_id}] Total raw chunks collected: {total_chunks}")
        
        # Deserialize indexes if they exist
        bm25_index = None
        faiss_index = None
        document_embeddings = None
        
        # Load BM25 index
        if chatbot_data.bm25_index_binary:
            logger.warning(f"[Request {request_id}] Loading BM25 index from binary data")
            try:
                bm25_index = pickle.loads(chatbot_data.bm25_index_binary)
                logger.warning(f"[Request {request_id}] BM25 index loaded successfully")
            except Exception as e:
                logger.error(f"[Request {request_id}] Error loading BM25 index: {e}")
        else:
            logger.warning(f"[Request {request_id}] No BM25 index binary data found")
        
        # Load FAISS index
        if chatbot_data.faiss_index_binary:
            logger.warning(f"[Request {request_id}] Loading FAISS index from binary data")
            try:
                logger.warning(f"[Request {request_id}] FAISS binary data size: {len(chatbot_data.faiss_index_binary)} bytes")
                logger.debug(f"[Request {request_id}] FAISS binary data preview: {chatbot_data.faiss_index_binary[:100]}")
                
                # Write binary to a temporary file
                with tempfile.NamedTemporaryFile(delete=False) as temp_index_file:
                    temp_index_file.write(chatbot_data.faiss_index_binary)
                    temp_index_path = temp_index_file.name
                logger.warning(f"[Request {request_id}] FAISS binary data written to temp file: {temp_index_path}")

                # Load it properly using FAISS
                faiss_index = faiss.read_index(temp_index_path)
                logger.warning(f"[Request {request_id}] FAISS index loaded successfully")

                # Clean up temp file
                os.remove(temp_index_path)
                logger.warning(f"[Request {request_id}] Temp file cleaned up")

            except Exception as e:
                logger.error(f"[Request {request_id}] Error loading FAISS index: {e}")
                logger.error(f"[Request {request_id}] FAISS loading traceback: {traceback.format_exc()}")
        else:
            logger.warning(f"[Request {request_id}] No FAISS index binary data found")
        
        # Get document embeddings
        logger.warning(f"[Request {request_id}] Processing document embeddings")
        embeddings = []
        for i, doc in enumerate(documents):
            if hasattr(doc, 'embedding') and doc.embedding and doc.embedding.embedding:
                embeddings.extend([doc.embedding.embedding])
                logger.warning(f"[Request {request_id}] Document {i+1}: Added embedding")
            else:
                logger.warning(f"[Request {request_id}] Document {i+1}: No embedding found")
        
        if embeddings:
            document_embeddings = np.array(embeddings)
            logger.warning(f"[Request {request_id}] Document embeddings array shape: {document_embeddings.shape}")
        else:
            logger.warning(f"[Request {request_id}] No document embeddings found")
        
        # Prepare tokenized docs
        logger.warning(f"[Request {request_id}] Preparing tokenized documents")
        tokenized_docs = []
        for i, doc in enumerate(documents):
            if doc.tokenized_text:
                token_count = len(doc.tokenized_text)
                tokenized_docs.extend(doc.tokenized_text)
                logger.warning(f"[Request {request_id}] Document {i+1}: Added {token_count} tokenized entries")
            else:
                logger.warning(f"[Request {request_id}] Document {i+1}: No tokenized text found")
        
        total_tokenized = len(tokenized_docs)
        logger.warning(f"[Request {request_id}] Total tokenized docs: {total_tokenized}")
        
        # Create result
        result = {
            'raw_chunks': raw_chunks,
            'bm25_index': bm25_index,
            'tokenized_docs': tokenized_docs,
            'faiss_index': faiss_index,
            'document_embeddings': document_embeddings,
            'intent_labels': chatbot_data.intent_labels or []
        }
        
        end_time = time.time()
        logger.warning(f"[Request {request_id}] prepare_document_data completed in {end_time - start_time:.3f}s")
        logger.warning(f"[Request {request_id}] Document data summary:")
        logger.warning(f"[Request {request_id}] - Raw chunks: {len(result['raw_chunks'])}")
        logger.warning(f"[Request {request_id}] - BM25 index: {'Available' if result['bm25_index'] else 'None'}")
        logger.warning(f"[Request {request_id}] - FAISS index: {'Available' if result['faiss_index'] else 'None'}")
        logger.warning(f"[Request {request_id}] - Document embeddings: {'Available' if result['document_embeddings'] is not None else 'None'}")
        logger.warning(f"[Request {request_id}] - Intent labels: {len(result['intent_labels'])}")
        
        return result
        
    except Exception as e:
        end_time = time.time()
        logger.error(f"[Request {request_id}] prepare_document_data failed after {end_time - start_time:.3f}s: {e}")
        logger.error(f"[Request {request_id}] prepare_document_data traceback: {traceback.format_exc()}")
        
        # Return empty structure on error
        return {
            'raw_chunks': [],
            'bm25_index': None,
            'tokenized_docs': None,
            'faiss_index': None,
            'document_embeddings': None,
            'intent_labels': []
        }