# api/services/nlp_manager.py
import nltk
from nltk.data import find
import spacy
from sentence_transformers import SentenceTransformer
from django.conf import settings
class NLPManager:
    _instance = None
    _nlp = None
    _embedding_model = None
    _resources_checked = False
    
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
            ('corpora/stopwords', 'stopwords')
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
    
    @property
    def nlp(self):
        self.ensure_resources()
        return self._nlp
    
    @property
    def embedding_model(self):
        self.ensure_resources()
        return self._embedding_model