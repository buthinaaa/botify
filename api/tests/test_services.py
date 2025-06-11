from django.test import TestCase
from unittest.mock import Mock, patch, MagicMock
from api.services.nlp_manager import NLPManager
import spacy


class NLPManagerTest(TestCase):
    
    def setUp(self):
        # Reset the singleton instance before each test
        NLPManager._instance = None
        NLPManager._nlp = None
        NLPManager._embedding_model = None
        NLPManager._resources_checked = False
        NLPManager._keybert_model = None
        NLPManager._ner_pipeline = None
        NLPManager._sentiment_model = None
        NLPManager._sentiment_tokenizer = None
        NLPManager._zero_shot_classifier = None
        NLPManager._response_model = None
        NLPManager._response_tokenizer = None
    
    def test_singleton_pattern(self):
        """Test that NLPManager follows singleton pattern"""
        instance1 = NLPManager.get_instance()
        instance2 = NLPManager.get_instance()
        
        self.assertIs(instance1, instance2)
        self.assertIsInstance(instance1, NLPManager)
    
    @patch('api.services.nlp_manager.nltk')
    @patch('api.services.nlp_manager.find')
    def test_ensure_nltk_resources_all_available(self, mock_find, mock_nltk):
        """Test NLTK resource checking when all resources are available"""
        # Mock that all resources are found
        mock_find.return_value = True
        
        manager = NLPManager.get_instance()
        manager._ensure_nltk_resources()
        
        # Should check for required resources
        expected_calls = 4  # Number of required resources
        self.assertEqual(mock_find.call_count, expected_calls)
        
        # Should not download anything since all are available
        mock_nltk.download.assert_not_called()
    
    @patch('api.services.nlp_manager.nltk')
    @patch('api.services.nlp_manager.find')
    def test_ensure_nltk_resources_missing_some(self, mock_find, mock_nltk):
        """Test NLTK resource checking when some resources are missing"""
        # Mock that some resources are missing
        def side_effect(resource_path):
            if 'punkt' in resource_path:
                raise LookupError("Resource not found")
            return True
        
        mock_find.side_effect = side_effect
        
        manager = NLPManager.get_instance()
        manager._ensure_nltk_resources()
        
        # Should download missing resources
        mock_nltk.download.assert_called()
    
    @patch('api.services.nlp_manager.spacy')
    @patch('api.services.nlp_manager.SentenceTransformer')
    @patch('api.services.nlp_manager.KeyBERT')
    @patch('api.services.nlp_manager.settings')
    @patch('api.services.nlp_manager.Path')
    @patch('api.services.nlp_manager.main_export')
    @patch('api.services.nlp_manager.AutoTokenizer')
    @patch('api.services.nlp_manager.ORTModelForSequenceClassification')
    @patch('api.services.nlp_manager.ORTModelForTokenClassification')
    @patch('api.services.nlp_manager.ORTModelForCausalLM')
    @patch('api.services.nlp_manager.pipeline')
    def test_load_models(self, mock_pipeline, mock_ort_causal, mock_ort_token, 
                        mock_ort_seq, mock_tokenizer, mock_export, mock_path,
                        mock_settings, mock_keybert, mock_sentence_transformer, mock_spacy):
        """Test model loading functionality"""
        # Setup mocks
        mock_settings.EMBEDDING_MODEL_NAME = 'test-embedding-model'
        mock_settings.QUANTIZED_MODELS_PATH = '/test/path'
        
        mock_nlp = Mock()
        mock_spacy.load.return_value = mock_nlp
        
        mock_embedding_model = Mock()
        mock_sentence_transformer.return_value = mock_embedding_model
        
        mock_keybert_model = Mock()
        mock_keybert.return_value = mock_keybert_model
        
        # Mock path operations
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True  # Assume models already exist
        mock_path_instance.mkdir.return_value = None
        mock_path.return_value = mock_path_instance
        
        # Mock model components
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_ort_seq.from_pretrained.return_value = mock_model_instance
        mock_ort_token.from_pretrained.return_value = mock_model_instance
        mock_ort_causal.from_pretrained.return_value = mock_model_instance
        
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance
        
        manager = NLPManager.get_instance()
        manager._load_models()
        
        # Verify models were loaded
        self.assertEqual(manager._nlp, mock_nlp)
        self.assertEqual(manager._embedding_model, mock_embedding_model)
        self.assertEqual(manager._keybert_model, mock_keybert_model)
        
        # Verify spacy model was loaded
        mock_spacy.load.assert_called_once_with("en_core_web_sm")
        
        # Verify sentence transformer was initialized
        mock_sentence_transformer.assert_called_once_with('test-embedding-model')
        
        # Verify KeyBERT was initialized
        mock_keybert.assert_called_once_with(model='test-embedding-model')
    
    @patch('api.services.nlp_manager.NLPManager._ensure_nltk_resources')
    @patch('api.services.nlp_manager.NLPManager._load_models')
    def test_ensure_resources_called_once(self, mock_load_models, mock_ensure_nltk):
        """Test that resources are only loaded once"""
        manager = NLPManager.get_instance()
        
        # Call ensure_resources multiple times
        manager.ensure_resources()
        manager.ensure_resources()
        manager.ensure_resources()
        
        # Should only be called once
        mock_ensure_nltk.assert_called_once()
        mock_load_models.assert_called_once()
        self.assertTrue(manager._resources_checked)
    
    @patch('api.services.nlp_manager.NLPManager.ensure_resources')
    def test_property_access_triggers_ensure_resources(self, mock_ensure_resources):
        """Test that accessing properties triggers resource ensuring"""
        manager = NLPManager.get_instance()
        
        # Mock the internal properties to avoid actual model loading
        manager._nlp = Mock()
        manager._embedding_model = Mock()
        manager._keybert_model = Mock()
        manager._ner_pipeline = Mock()
        manager._sentiment_model = Mock()
        manager._sentiment_tokenizer = Mock()
        manager._zero_shot_classifier = Mock()
        manager._response_model = Mock()
        manager._response_tokenizer = Mock()
        
        # Access each property
        _ = manager.nlp
        _ = manager.embedding_model
        _ = manager.kw_model
        _ = manager.ner_pipeline
        _ = manager.sentiment_model
        _ = manager.sentiment_tokenizer
        _ = manager.zero_shot_classifier
        _ = manager.response_model
        _ = manager.response_tokenizer
        
        # Should call ensure_resources for each property access
        self.assertEqual(mock_ensure_resources.call_count, 9)
    
    def test_multiple_instances_are_same(self):
        """Test that multiple calls to get_instance return the same object"""
        instances = [NLPManager.get_instance() for _ in range(5)]
        
        # All instances should be the same object
        for instance in instances[1:]:
            self.assertIs(instances[0], instance)
    
    @patch('api.services.nlp_manager.NLPManager._load_models')
    @patch('api.services.nlp_manager.NLPManager._ensure_nltk_resources')
    def test_resources_checked_flag(self, mock_ensure_nltk, mock_load_models):
        """Test that _resources_checked flag works correctly"""
        manager = NLPManager.get_instance()
        
        # Initially should be False
        self.assertFalse(manager._resources_checked)
        
        # After calling ensure_resources, should be True
        manager.ensure_resources()
        self.assertTrue(manager._resources_checked)
        
        # Calling again should not trigger loading again
        mock_ensure_nltk.reset_mock()
        mock_load_models.reset_mock()
        
        manager.ensure_resources()
        mock_ensure_nltk.assert_not_called()
        mock_load_models.assert_not_called() 