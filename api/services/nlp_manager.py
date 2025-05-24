# api/services/nlp_manager.py
import nltk
from nltk.data import find
import spacy
from sentence_transformers import SentenceTransformer
from django.conf import settings
from keybert import KeyBERT
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForTokenClassification,
    pipeline
)
import torch

class NLPManager:
    _instance = None
    _nlp = None
    _embedding_model = None
    _resources_checked = False
    _keybert_model = None
    
    # Add new model properties
    _ner_pipeline = None
    _sentiment_model = None
    _sentiment_tokenizer = None
    _zero_shot_classifier = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = NLPManager()
        return cls._instance
    
    def ensure_resources(self):
        """Ensures all NLP resources are available and models are loaded."""
        if not self._resources_checked:
            self._ensure_nltk_resources()
            self._load_models()
            self._resources_checked = True
    
    def _ensure_nltk_resources(self):
        """Ensures NLTK resources are available, downloading only if needed."""
        required_resources = [
            ('tokenizers/punkt', 'punkt'),
            ('tokenizers/punkt_tab', 'punkt_tab'), 
            ('corpora/stopwords', 'stopwords'),
            ('corpora/wordnet', 'wordnet')
        ]
        
        for resource_path, resource_name in required_resources:
            try:
                find(resource_path)
                print(f"NLTK resource '{resource_name}' is already available.")
            except LookupError:
                print(f"Downloading NLTK resource '{resource_name}'...")
                nltk.download(resource_name)
    
    def _load_models(self):
        """Load NLP models once."""
        print("Loading spaCy model...")
        self._nlp = spacy.load("en_core_web_sm")
        
        print("Loading sentence transformer model...")
        self._embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        self._keybert_model = KeyBERT(model=settings.EMBEDDING_MODEL_NAME)
        
        # Load NER model
        print("Loading NER model...")
        ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER-uncased")
        ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER-uncased")
        self._ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")
        
        # Load sentiment model
        print("Loading sentiment model...")
        self._sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        self._sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        self._sentiment_model.eval()
        
        # Load zero-shot classification model
        print("Loading zero-shot classification model...")
        self._zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    @property
    def nlp(self):
        self.ensure_resources()
        return self._nlp
    
    @property
    def embedding_model(self):
        self.ensure_resources()
        return self._embedding_model
    
    @property
    def kw_model(self):
        self.ensure_resources()
        return self._keybert_model
    
    @property
    def ner_pipeline(self):
        self.ensure_resources()
        return self._ner_pipeline
    
    @property
    def sentiment_model(self):
        self.ensure_resources()
        return self._sentiment_model
    
    @property
    def sentiment_tokenizer(self):
        self.ensure_resources()
        return self._sentiment_tokenizer
    
    @property
    def zero_shot_classifier(self):
        self.ensure_resources()
        return self._zero_shot_classifier
    