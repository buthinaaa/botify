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
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    pipeline
)
from optimum.onnxruntime import ORTModelForSequenceClassification,ORTModelForTokenClassification,ORTModelForCausalLM
from pathlib import Path
from optimum.exporters.onnx import main_export
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
    _response_model = None
    _response_tokenizer = None
    
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

        def maybe_export(model_id, save_dir, task):
            model_path = save_dir / "model.onnx"
            if not model_path.exists():
                print(f"Exporting and quantizing model: {model_id}")
                main_export(
                    model_name_or_path=model_id,
                    output=save_dir,
                    task=task,
                    quantization="dynamic"
                )
            else:
                print(f"Using cached quantized model: {model_id}")

        # Load NER model
        print("Loading NER model...")
        ner_model_id = "dslim/bert-base-NER-uncased"
        ner_save_dir = Path("quantized_models/ner")
        ner_save_dir.mkdir(parents=True, exist_ok=True)
        maybe_export(ner_model_id, ner_save_dir, "token-classification")
        ner_tokenizer = AutoTokenizer.from_pretrained(ner_save_dir)
        ner_ort_model = ORTModelForTokenClassification.from_pretrained(ner_save_dir)
        self._ner_pipeline = pipeline(
            "ner",
            model=ner_ort_model,
            tokenizer=ner_tokenizer,
            aggregation_strategy="simple",
        )

        # Sentiment model
        print("Loading sentiment model...")
        sentiment_model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        sentiment_save_dir = Path("quantized_models/sentiment")
        sentiment_save_dir.mkdir(parents=True, exist_ok=True)
        maybe_export(sentiment_model_id, sentiment_save_dir, "text-classification")
        self._sentiment_model = ORTModelForSequenceClassification.from_pretrained(sentiment_save_dir)
        self._sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_save_dir)

        # Zero-shot classification model
        print("Loading zero-shot classification model...")
        zero_shot_model_id = "facebook/bart-large-mnli"
        zero_shot_save_dir = Path("quantized_models/zero_shot")
        zero_shot_save_dir.mkdir(parents=True, exist_ok=True)
        maybe_export(zero_shot_model_id, zero_shot_save_dir, "zero-shot-classification")
        zero_shot_tokenizer = AutoTokenizer.from_pretrained(zero_shot_save_dir)
        zero_shot_model = ORTModelForSequenceClassification.from_pretrained(zero_shot_save_dir)
        self._zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model=zero_shot_model,
            tokenizer=zero_shot_tokenizer,
        )

        # Response model
        print("Loading response model...")
        response_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        response_save_dir = Path("quantized_models/response_generation")
        response_save_dir.mkdir(parents=True, exist_ok=True)
        maybe_export(response_model_id, response_save_dir, "text-generation")
        self._response_model = ORTModelForCausalLM.from_pretrained(response_save_dir, use_cache=False, use_io_binding=False)
        self._response_tokenizer = AutoTokenizer.from_pretrained(response_save_dir)

    
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
    
    @property
    def response_model(self):
        self.ensure_resources()
        return self._response_model
    
    @property
    def response_tokenizer(self):
        self.ensure_resources()
        return self._response_tokenizer
    