import tempfile
from django.db import transaction
from rest_framework import serializers
from api.models.chatbot_models import Chatbot, ChatbotData, ChatbotDocument, DocumentEmbedding
from api.services.docs_analyser import process_uploaded_files
from api.utilities.files_processing import clean_temp_file, save_file_to_temp_dir
import faiss
import pickle
import os

class ChatbotSerializer(serializers.ModelSerializer):
    documents = serializers.ListField(
        child=serializers.FileField(max_length=10 * 1024 * 1024, 
                                   allow_empty_file=False, 
                                   use_url=True),
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
        
            processed_docs, bm25_index, tokenized_docs, faiss_index, document_embeddings, intent_labels = process_uploaded_files(temp_file_paths)
            with transaction.atomic():
                chatbot = Chatbot.objects.create(user=user, **validated_data)
                chatbot_data = ChatbotData.objects.create(
                    chatbot=chatbot,
                    intent_labels=intent_labels,
                )
                temp_index_file = tempfile.NamedTemporaryFile(delete=False)
                temp_index_path = temp_index_file.name
                temp_index_file.close()  # Close it so FAISS can write to it

                try:
                    # Write the FAISS index to the temporary file
                    faiss.write_index(faiss_index, temp_index_path)
                    
                    # Read the file content into memory
                    with open(temp_index_path, 'rb') as file:
                        index_binary = file.read()
                    
                    # Store the binary data in your database
                    chatbot_data.faiss_index_binary = index_binary
                    
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_index_path):
                        os.remove(temp_index_path)

                # Store the BM25 index as pickled binary data
                chatbot_data.bm25_index_binary = pickle.dumps(bm25_index)
                chatbot_data.save()

                for i in range(len(processed_docs)):
                    filename = os.path.basename(temp_file_paths[i])
                    
                    document = ChatbotDocument.objects.create(
                        chatbot_data=chatbot_data,
                        original_filename=filename,
                        processed_text=processed_docs[i],
                        tokenized_text=tokenized_docs[i]
                    )
                    
                    embedding_vector = document_embeddings[i]
                    
                    DocumentEmbedding.objects.create(
                        document=document,
                        embedding=embedding_vector.tolist()
                    )
            return chatbot
        finally:
            for path in temp_file_paths:
                clean_temp_file(path)
        return None