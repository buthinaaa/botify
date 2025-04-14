from django.db import models
from django.contrib.auth.models import User
from django.conf import settings
from django.contrib.postgres.fields import ArrayField

class Chatbot(models.Model):
    name = models.CharField(max_length=100)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chatbots')
    
    def __str__(self):
        return self.name

class ChatbotData(models.Model):
    chatbot = models.OneToOneField(Chatbot, on_delete=models.CASCADE, related_name='data', null=True)
   
    # For binary indices (FAISS and BM25)
    faiss_index_binary = models.BinaryField(null=True)
    bm25_index_binary = models.BinaryField(null=True)
    embedding_model_name = models.CharField(max_length=100, default=settings.EMBEDDING_MODEL_NAME)
    intent_labels = ArrayField(models.CharField(max_length=100, null=True), null=True)
    def __str__(self):
        return f"Data for {self.chatbot.name}"



class ChatbotDocument(models.Model):
    chatbot_data = models.ForeignKey(ChatbotData, on_delete=models.CASCADE, related_name='documents')
    original_filename = models.CharField(max_length=255, null=True)
    processed_text = models.TextField(null=True)
    
    tokenized_text = ArrayField(models.CharField(max_length=100, null=True), null=True)
    
    def __str__(self):
        return f"Document: {self.original_filename}"
    
class DocumentEmbedding(models.Model):
    document = models.OneToOneField(ChatbotDocument, on_delete=models.CASCADE, related_name='embedding')
    embedding = ArrayField(models.FloatField(), null=True)
    
    def __str__(self):
        return f"Embedding for {self.document.original_filename}"