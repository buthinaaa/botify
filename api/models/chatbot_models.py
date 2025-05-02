from django.db import models
from django.contrib.auth.models import User
from django.conf import settings
from django.contrib.postgres.fields import ArrayField
import uuid
SENDER_CHOICES = [
        ("user", "User"),
        ("bot", "Bot"),
        ("admin", "Admin"),
    ]

class Chatbot(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chatbots')
    
    def __str__(self):
        return self.name

class ChatbotData(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    chatbot = models.OneToOneField(Chatbot, on_delete=models.CASCADE, related_name='data', null=True)
    faiss_index_binary = models.BinaryField(null=True)
    bm25_index_binary = models.BinaryField(null=True)
    embedding_model_name = models.CharField(max_length=100, default=settings.EMBEDDING_MODEL_NAME)
    intent_labels = ArrayField(models.CharField(max_length=100, null=True), null=True)
    def __str__(self):
        return f"Data for {self.chatbot.name}"

class ChatbotDocument(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    chatbot_data = models.ForeignKey(ChatbotData, on_delete=models.CASCADE, related_name='documents')
    original_filename = models.CharField(max_length=255, null=True)
    chunks = chunks = ArrayField(models.TextField(), null=True)
    tokenized_text = ArrayField(models.CharField(max_length=100, null=True), null=True)
    
    def __str__(self):
        return f"Document: {self.original_filename}"
    
class DocumentEmbedding(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.OneToOneField(ChatbotDocument, on_delete=models.CASCADE, related_name='embedding')
    embedding = ArrayField(models.FloatField(), null=True)
    
    def __str__(self):
        return f"Embedding for {self.document.original_filename}"
    
class ChatSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    chatbot = models.ForeignKey(Chatbot, on_delete=models.CASCADE, related_name='chat_sessions')
    created_at = models.DateTimeField(auto_now_add=True)
    session_id = models.CharField(max_length=255, null=True, blank=True)

class Message(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    chatbot = models.ForeignKey(Chatbot, on_delete=models.CASCADE, related_name="messages")
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="messages")
    sender = models.CharField(max_length=10, choices=SENDER_CHOICES)
    original_text = models.TextField(null=True, blank=True)  # New field to store original text
    processed_text_use = models.TextField(null=True, blank=True) 
    timestamp = models.DateTimeField(auto_now_add=True)