from django.test import TestCase
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.db import IntegrityError
from api.models.chatbot_models import (
    Chatbot, ChatbotData, ChatbotDocument, DocumentEmbedding, 
    ChatSession, Message, SENDER_CHOICES
)
from api.models.user_models import CustomUser
import uuid

User = get_user_model()

class CustomUserModelTest(TestCase):
    def setUp(self):
        self.user_data = {
            'username': 'testuser',
            'email': 'test@example.com',
            'phone_number': '+1234567890',
            'first_name': 'Test',
            'last_name': 'User'
        }
    
    def test_create_user(self):
        """Test creating a user with valid data"""
        user = CustomUser.objects.create_user(
            username=self.user_data['username'],
            email=self.user_data['email'],
            phone_number=self.user_data['phone_number'],
            password='testpass123'
        )
        self.assertEqual(user.username, 'testuser')
        self.assertEqual(user.email, 'test@example.com')
        self.assertEqual(str(user.phone_number), '+1234567890')
        self.assertTrue(user.check_password('testpass123'))
    
    def test_unique_phone_number(self):
        """Test that phone numbers must be unique"""
        CustomUser.objects.create_user(
            username='user1',
            email='user1@example.com',
            phone_number='+1234567890',
            password='testpass123'
        )
        
        with self.assertRaises(IntegrityError):
            CustomUser.objects.create_user(
                username='user2',
                email='user2@example.com',
                phone_number='+1234567890',  # Duplicate phone number
                password='testpass123'
            )
    
    def test_str_representation(self):
        """Test string representation of user"""
        user = CustomUser.objects.create_user(
            username='testuser',
            email='test@example.com',
            phone_number='+1234567890'
        )
        self.assertEqual(str(user), 'testuser')

class ChatbotModelTest(TestCase):
    def setUp(self):
        self.user = CustomUser.objects.create_user(
            username='testuser',
            email='test@example.com',
            phone_number='+1234567890',
            password='testpass123'
        )
        self.chatbot_data = {
            'name': 'Test Chatbot',
            'user': self.user,
            'primary_color': '#FF0000',
            'text_color': '#000000',
            'welcome_message': 'Welcome to our chatbot!',
            'welcome_popup': 'Hello! How can I help you today?',
            'chat_input': 'Type your message here...'
        }
    
    def test_create_chatbot(self):
        """Test creating a chatbot with valid data"""
        chatbot = Chatbot.objects.create(**self.chatbot_data)
        self.assertEqual(chatbot.name, 'Test Chatbot')
        self.assertEqual(chatbot.user, self.user)
        self.assertEqual(chatbot.primary_color, '#FF0000')
        self.assertEqual(chatbot.text_color, '#000000')
        self.assertIsInstance(chatbot.id, uuid.UUID)
    
    def test_chatbot_str_representation(self):
        """Test string representation of chatbot"""
        chatbot = Chatbot.objects.create(**self.chatbot_data)
        self.assertEqual(str(chatbot), 'Test Chatbot')
    
    def test_chatbot_defaults(self):
        """Test default values for chatbot fields"""
        chatbot = Chatbot.objects.create(
            name='Simple Bot',
            user=self.user
        )
        self.assertEqual(chatbot.primary_color, '#000000')
        self.assertEqual(chatbot.text_color, '#000000')
        self.assertIsNone(chatbot.logo)
        self.assertIsNone(chatbot.welcome_message)
    
    def test_chatbot_user_relationship(self):
        """Test relationship between chatbot and user"""
        chatbot1 = Chatbot.objects.create(name='Bot 1', user=self.user)
        chatbot2 = Chatbot.objects.create(name='Bot 2', user=self.user)
        
        user_chatbots = self.user.chatbots.all()
        self.assertEqual(user_chatbots.count(), 2)
        self.assertIn(chatbot1, user_chatbots)
        self.assertIn(chatbot2, user_chatbots)

class ChatbotDataModelTest(TestCase):
    def setUp(self):
        self.user = CustomUser.objects.create_user(
            username='testuser',
            email='test@example.com',
            phone_number='+1234567890'
        )
        self.chatbot = Chatbot.objects.create(
            name='Test Bot',
            user=self.user
        )
    
    def test_create_chatbot_data(self):
        """Test creating chatbot data"""
        data = ChatbotData.objects.create(
            chatbot=self.chatbot,
            embedding_model_name='test-model',
            intent_labels=['greeting', 'goodbye', 'question']
        )
        self.assertEqual(data.chatbot, self.chatbot)
        self.assertEqual(data.embedding_model_name, 'test-model')
        self.assertEqual(data.intent_labels, ['greeting', 'goodbye', 'question'])
        self.assertIsInstance(data.id, uuid.UUID)
    
    def test_chatbot_data_str_representation(self):
        """Test string representation of chatbot data"""
        data = ChatbotData.objects.create(chatbot=self.chatbot)
        self.assertEqual(str(data), 'Data for Test Bot')
    
    def test_one_to_one_relationship(self):
        """Test one-to-one relationship between chatbot and data"""
        data = ChatbotData.objects.create(chatbot=self.chatbot)
        
        # Test accessing data from chatbot
        self.assertEqual(self.chatbot.data, data)
        
        # Test that creating another data for same chatbot raises error
        with self.assertRaises(IntegrityError):
            ChatbotData.objects.create(chatbot=self.chatbot)

class ChatbotDocumentModelTest(TestCase):
    def setUp(self):
        self.user = CustomUser.objects.create_user(
            username='testuser',
            email='test@example.com',
            phone_number='+1234567890'
        )
        self.chatbot = Chatbot.objects.create(name='Test Bot', user=self.user)
        self.chatbot_data = ChatbotData.objects.create(chatbot=self.chatbot)
    
    def test_create_document(self):
        """Test creating a chatbot document"""
        document = ChatbotDocument.objects.create(
            chatbot_data=self.chatbot_data,
            original_filename='test_document.pdf',
            chunks=['chunk1', 'chunk2', 'chunk3'],
            tokenized_text=['token1', 'token2', 'token3']
        )
        self.assertEqual(document.chatbot_data, self.chatbot_data)
        self.assertEqual(document.original_filename, 'test_document.pdf')
        self.assertEqual(document.chunks, ['chunk1', 'chunk2', 'chunk3'])
        self.assertEqual(document.tokenized_text, ['token1', 'token2', 'token3'])
    
    def test_document_str_representation(self):
        """Test string representation of document"""
        document = ChatbotDocument.objects.create(
            chatbot_data=self.chatbot_data,
            original_filename='test.pdf'
        )
        self.assertEqual(str(document), 'Document: test.pdf')

class DocumentEmbeddingModelTest(TestCase):
    def setUp(self):
        self.user = CustomUser.objects.create_user(
            username='testuser',
            email='test@example.com',
            phone_number='+1234567890'
        )
        self.chatbot = Chatbot.objects.create(name='Test Bot', user=self.user)
        self.chatbot_data = ChatbotData.objects.create(chatbot=self.chatbot)
        self.document = ChatbotDocument.objects.create(
            chatbot_data=self.chatbot_data,
            original_filename='test.pdf'
        )
    
    def test_create_embedding(self):
        """Test creating document embedding"""
        embedding = DocumentEmbedding.objects.create(
            document=self.document,
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        self.assertEqual(embedding.document, self.document)
        self.assertEqual(embedding.embedding, [0.1, 0.2, 0.3, 0.4, 0.5])
    
    def test_embedding_str_representation(self):
        """Test string representation of embedding"""
        embedding = DocumentEmbedding.objects.create(
            document=self.document,
            embedding=[0.1, 0.2, 0.3]
        )
        self.assertEqual(str(embedding), 'Embedding for test.pdf')

class ChatSessionModelTest(TestCase):
    def setUp(self):
        self.user = CustomUser.objects.create_user(
            username='testuser',
            email='test@example.com',
            phone_number='+1234567890'
        )
        self.chatbot = Chatbot.objects.create(name='Test Bot', user=self.user)
    
    def test_create_chat_session(self):
        """Test creating a chat session"""
        session = ChatSession.objects.create(
            chatbot=self.chatbot,
            session_id='session_123',
            is_intervened=False
        )
        self.assertEqual(session.chatbot, self.chatbot)
        self.assertEqual(session.session_id, 'session_123')
        self.assertFalse(session.is_intervened)
        self.assertIsNotNone(session.created_at)
        self.assertIsInstance(session.id, uuid.UUID)
    
    def test_chat_session_defaults(self):
        """Test default values for chat session"""
        session = ChatSession.objects.create(chatbot=self.chatbot)
        self.assertFalse(session.is_intervened)
        self.assertIsNone(session.session_id)

class MessageModelTest(TestCase):
    def setUp(self):
        self.user = CustomUser.objects.create_user(
            username='testuser',
            email='test@example.com',
            phone_number='+1234567890'
        )
        self.chatbot = Chatbot.objects.create(name='Test Bot', user=self.user)
        self.session = ChatSession.objects.create(chatbot=self.chatbot)
    
    def test_create_message(self):
        """Test creating a message"""
        message = Message.objects.create(
            chatbot=self.chatbot,
            session=self.session,
            sender='user',
            original_text='Hello, how are you?',
            processed_text_use='hello how are you',
            sentiment='positive',
            overall_sentiment={'positive': 0.8, 'negative': 0.1, 'neutral': 0.1},
            ner_entities=['greeting'],
            intent=['greeting']
        )
        self.assertEqual(message.chatbot, self.chatbot)
        self.assertEqual(message.session, self.session)
        self.assertEqual(message.sender, 'user')
        self.assertEqual(message.original_text, 'Hello, how are you?')
        self.assertEqual(message.processed_text_use, 'hello how are you')
        self.assertEqual(message.sentiment, 'positive')
        self.assertIsNotNone(message.timestamp)
    
    def test_sender_choices(self):
        """Test that sender field validates against choices"""
        # Valid choices should work
        for sender_choice, _ in SENDER_CHOICES:
            message = Message.objects.create(
                chatbot=self.chatbot,
                session=self.session,
                sender=sender_choice,
                original_text='Test message'
            )
            self.assertEqual(message.sender, sender_choice)
            message.delete()  # Clean up for next iteration
    
    def test_message_relationships(self):
        """Test message relationships with chatbot and session"""
        message1 = Message.objects.create(
            chatbot=self.chatbot,
            session=self.session,
            sender='user',
            original_text='Message 1'
        )
        message2 = Message.objects.create(
            chatbot=self.chatbot,
            session=self.session,
            sender='bot',
            original_text='Message 2'
        )
        
        # Test chatbot messages
        chatbot_messages = self.chatbot.messages.all()
        self.assertEqual(chatbot_messages.count(), 2)
        self.assertIn(message1, chatbot_messages)
        self.assertIn(message2, chatbot_messages)
        
        # Test session messages
        session_messages = self.session.messages.all()
        self.assertEqual(session_messages.count(), 2)
        self.assertIn(message1, session_messages)
        self.assertIn(message2, session_messages)
    
    def test_optional_fields(self):
        """Test that optional fields can be None"""
        message = Message.objects.create(
            chatbot=self.chatbot,
            session=self.session,
            sender='user'
        )
        self.assertIsNone(message.original_text)
        self.assertIsNone(message.processed_text_use)
        self.assertIsNone(message.sentiment)
        self.assertIsNone(message.overall_sentiment)
        self.assertIsNone(message.ner_entities)
        self.assertIsNone(message.intent) 