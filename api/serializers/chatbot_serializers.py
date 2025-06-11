import tempfile
from django.db import transaction
from rest_framework import serializers
from api.models.chatbot_models import Chatbot, ChatbotData, ChatbotDocument, DocumentEmbedding
from api.services.docs_analyzer import process_uploaded_files
from api.utilities.files_processing import clean_temp_file, save_file_to_temp_dir
import faiss
import pickle
import os
from api.services.docs_analyzer import extract_text_from_pdf, extract_text_from_txt, extract_text_from_docx, semantic_window_chunking
import numpy as np
from rest_framework.response import Response

class ChatbotSerializer(serializers.ModelSerializer):
    def validate_file_type(file):
        allowed_extensions = ['.pdf', '.txt', '.docx']
        ext = os.path.splitext(file.name)[1].lower()
        if ext not in allowed_extensions:
            raise serializers.ValidationError(f'File type not allowed. Only PDF, TXT, and DOCX files are accepted.')

    documents = serializers.ListField(
        child=serializers.FileField(
            max_length=10 * 1024 * 1024, 
            allow_empty_file=False,
            use_url=True,
            validators=[validate_file_type]
        ),
        write_only=True,
        required=True
    )
    class Meta:
        model = Chatbot
        exclude = ['user']
    
    def create(self, validated_data):
        user = self.context['request'].user
        documents = validated_data.pop('documents')
        temp_file_paths = []
        
        try:
            for document in documents:
                file_path = save_file_to_temp_dir(document)
                temp_file_paths.append(str(file_path))
        
            raw_chunks, bm25_index, tokenized_docs, faiss_index, document_embeddings, intent_labels = process_uploaded_files(temp_file_paths)
            chunk_counts = []
            for path in temp_file_paths:
                if path.endswith(".pdf"):
                    text = extract_text_from_pdf(path)
                elif path.endswith(".txt"):
                    text = extract_text_from_txt(path)
                elif path.endswith(".docx"):
                    text = extract_text_from_docx(path)
                else:
                    raise serializers.ValidationError(f"Unsupported file type: {path}")
                
                chunks = semantic_window_chunking(text)
                chunk_counts.append(len(chunks))

            with transaction.atomic():
                chatbot = Chatbot.objects.create(user=user, **validated_data)
                chatbot_data = ChatbotData.objects.create(
                    chatbot=chatbot,
                    intent_labels=intent_labels,
                )
                temp_index_file = tempfile.NamedTemporaryFile(delete=False)
                temp_index_path = temp_index_file.name
                temp_index_file.close()
                try:
                    faiss.write_index(faiss_index, temp_index_path)
                    with open(temp_index_path, 'rb') as file:
                        index_binary = file.read()
                    chatbot_data.faiss_index_binary = index_binary
                    
                finally:
                    if os.path.exists(temp_index_path):
                        os.remove(temp_index_path)

                chatbot_data.bm25_index_binary = pickle.dumps(bm25_index)
                chatbot_data.save()
                chunk_pointer = 0
                for i, file_path in enumerate(temp_file_paths):
                    filename = os.path.basename(temp_file_paths[i])
                    num_chunks = chunk_counts[i]
                    chunk_embeds = document_embeddings[chunk_pointer:chunk_pointer + num_chunks]
                    avg_embedding = np.mean(chunk_embeds, axis=0)
                    tokenized_text = tokenized_docs[i]

                    document = ChatbotDocument.objects.create(
                        chatbot_data=chatbot_data,
                        original_filename=filename,
                        chunks=raw_chunks[chunk_pointer:chunk_pointer + num_chunks],
                        tokenized_text=tokenized_text
                    )
                    
                    DocumentEmbedding.objects.create(
                        document=document,
                        embedding=avg_embedding.tolist()
                    )

                    chunk_pointer += num_chunks

            return chatbot
        finally:
            for path in temp_file_paths:
                clean_temp_file(path)
        return None