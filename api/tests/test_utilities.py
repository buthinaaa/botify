from django.test import TestCase
from unittest.mock import Mock, patch, mock_open
from api.utilities.files_processing import save_file_to_temp_dir, clean_temp_file
from api.utilities.message_processing import (
    normalize_repeated_chars, preprocess_message
)
from api.utilities.sentiment_analysis import (
    get_overall_sentiment, update_context_with_sentiment,
    check_for_fallback
)
import tempfile
import os
import uuid
from pathlib import Path
from collections import Counter


class FilesProcessingTest(TestCase):
    
    def test_save_file_to_temp_dir(self):
        """Test saving file to temporary directory"""
        mock_file = Mock()
        mock_file.name = 'test_document.pdf'
        mock_file.chunks.return_value = [b'chunk1', b'chunk2', b'chunk3']
        
        with patch('api.utilities.files_processing.tempfile.gettempdir') as mock_temp:
            mock_temp.return_value = '/tmp'
            with patch('builtins.open', mock_open()) as mock_file_open:
                with patch('os.makedirs') as mock_makedirs:
                    with patch('uuid.uuid4') as mock_uuid:
                        mock_uuid.return_value = uuid.UUID('12345678-1234-5678-1234-567812345678')
                        
                        result = save_file_to_temp_dir(mock_file)
                        
                        expected_path = Path('/tmp/chatbot_uploads/12345678-1234-5678-1234-567812345678.pdf')
                        self.assertEqual(result, expected_path)
                        mock_makedirs.assert_called_once_with(Path('/tmp/chatbot_uploads'), exist_ok=True)
                        mock_file_open.assert_called_once_with(expected_path, 'wb')
    
    def test_clean_temp_file_success(self):
        """Test successful cleanup of temporary file"""
        test_path = '/tmp/test_file.txt'
        
        with patch('os.remove') as mock_remove:
            clean_temp_file(test_path)
            mock_remove.assert_called_once_with(test_path)
    
    def test_clean_temp_file_failure(self):
        """Test cleanup when file removal fails"""
        test_path = '/tmp/test_file.txt'
        
        with patch('os.remove', side_effect=OSError('File not found')):
            with patch('builtins.print') as mock_print:
                clean_temp_file(test_path)
                mock_print.assert_called_once_with(f"Failed to delete temporary file: {test_path}")


class MessageProcessingTest(TestCase):
    
    def test_normalize_repeated_chars(self):
        """Test normalization of repeated characters"""
        test_cases = [
            ('sooooo happy', 'soo happy'),
            ('hellooooo', 'helloo'),
            ('wowwwwww', 'woww'),
            ('normal text', 'normal text'),
            ('aaaaabbbbccccc', 'aabbcc'),
            ('', ''),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = normalize_repeated_chars(input_text)
                self.assertEqual(result, expected)
    
    @patch('api.utilities.message_processing.SpellChecker')
    @patch('api.utilities.message_processing.WordNetLemmatizer')
    @patch('api.utilities.message_processing.stopwords')
    def test_preprocess_message_basic(self, mock_stopwords, mock_lemmatizer, mock_spellchecker):
        """Test basic message preprocessing"""
        # Setup mocks
        mock_stopwords.words.return_value = ['the', 'is', 'and', 'or']
        
        mock_lemmatizer_instance = Mock()
        mock_lemmatizer_instance.lemmatize.side_effect = lambda word, pos=None: word
        mock_lemmatizer.return_value = mock_lemmatizer_instance
        
        mock_spell_instance = Mock()
        mock_spell_instance.correction.side_effect = lambda word: word
        mock_spellchecker.return_value = mock_spell_instance
        
        message = "Hello! How are you doing today? 😊"
        context = []
        
        result = preprocess_message(message, context)
        
        # Check that all expected keys are present
        expected_keys = ['original', 'clean_text', 'text_use', 'tokens', 
                        'lemmatized_tokens', 'corrected_tokens', 'emojis']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check original is preserved
        self.assertEqual(result['original'], message)
        
        # Check that context was updated
        self.assertEqual(len(context), 1)
        self.assertEqual(context[0]['user'], message)
    
    @patch('api.utilities.message_processing.SpellChecker')
    @patch('api.utilities.message_processing.WordNetLemmatizer')
    @patch('api.utilities.message_processing.stopwords')
    def test_preprocess_message_error_handling(self, mock_stopwords, mock_lemmatizer, mock_spellchecker):
        """Test preprocessing with error handling"""
        # Setup mocks to raise an exception
        mock_stopwords.words.side_effect = Exception('NLTK error')
        
        message = "Test message"
        
        result = preprocess_message(message)
        
        # Should return basic structure on error
        self.assertEqual(result['original'], message)
        self.assertEqual(result['clean_text'], 'test message')
        self.assertEqual(result['text_use'], 'test message')
        self.assertIsInstance(result['tokens'], list)
        self.assertIsInstance(result['emojis'], list)
    
    def test_preprocess_message_emoji_detection(self):
        """Test emoji detection in preprocessing"""
        with patch('api.utilities.message_processing.SpellChecker'), \
             patch('api.utilities.message_processing.WordNetLemmatizer'), \
             patch('api.utilities.message_processing.stopwords'):
            
            message = "I'm happy 😊😍 today!"
            
            # Mock emoji.EMOJI_DATA
            with patch('api.utilities.message_processing.emoji.EMOJI_DATA', {'😊': None, '😍': None}):
                result = preprocess_message(message)
                
                # Should detect both emojis
                self.assertEqual(len(result['emojis']), 2)
                self.assertIn('😊', result['emojis'])
                self.assertIn('😍', result['emojis'])


class SentimentAnalysisTest(TestCase):
    
    def test_get_overall_sentiment_basic(self):
        """Test basic overall sentiment calculation"""
        context = [
            {'sentiment': 'positive'},
            {'sentiment': 'positive'},
            {'sentiment': 'negative'},
            {'sentiment': 'neutral'},
            {'sentiment': 'positive'}
        ]
        
        result = get_overall_sentiment(context, window=5)
        expected = Counter(['positive', 'positive', 'negative', 'neutral', 'positive'])
        
        self.assertEqual(result, expected)
        self.assertEqual(result['positive'], 3)
        self.assertEqual(result['negative'], 1)
        self.assertEqual(result['neutral'], 1)
    
    def test_get_overall_sentiment_with_window(self):
        """Test overall sentiment with limited window"""
        context = [
            {'sentiment': 'negative'},  # Should be ignored (outside window)
            {'sentiment': 'negative'},  # Should be ignored (outside window)
            {'sentiment': 'positive'},  # Last 3
            {'sentiment': 'positive'},  # Last 3
            {'sentiment': 'negative'},  # Last 3
        ]
        
        result = get_overall_sentiment(context, window=3)
        
        self.assertEqual(result['positive'], 2)
        self.assertEqual(result['negative'], 1)
        self.assertNotIn('neutral', result)  # Should not be present
    
    def test_get_overall_sentiment_empty_context(self):
        """Test overall sentiment with empty context"""
        result = get_overall_sentiment([], window=5)
        self.assertEqual(result, Counter())
    
    def test_get_overall_sentiment_none_context(self):
        """Test overall sentiment with None context"""
        result = get_overall_sentiment(None, window=5)
        self.assertEqual(result, Counter())
    
    def test_get_overall_sentiment_missing_sentiment(self):
        """Test overall sentiment with entries missing sentiment"""
        context = [
            {'sentiment': 'positive'},
            {'user': 'hello'},  # No sentiment field
            {'sentiment': 'negative'},
            {'message': 'test'},  # No sentiment field
        ]
        
        result = get_overall_sentiment(context, window=10)
        
        self.assertEqual(result['positive'], 1)
        self.assertEqual(result['negative'], 1)
        self.assertEqual(len(result), 2)  # Only 2 valid sentiments
    
    def test_update_context_with_sentiment(self):
        """Test updating context with sentiment"""
        context = [
            {'user': 'previous message', 'sentiment': 'neutral'}
        ]
        
        preprocessed_message = {
            'original': 'Hello world',
            'clean_text': 'hello world'
        }
        
        update_context_with_sentiment(context, preprocessed_message, 'positive')
        
        self.assertEqual(len(context), 2)
        self.assertEqual(context[1]['user'], 'Hello world')
        self.assertEqual(context[1]['sentiment'], 'positive')
    
    def test_check_for_fallback_insufficient_messages(self):
        """Test fallback check with insufficient messages"""
        context = [
            {'sentiment': 'negative'},
            {'sentiment': 'negative'}
        ]
        
        overall_counts = Counter({'negative': 2})
        
        # Should not trigger fallback due to insufficient messages
        result = check_for_fallback(
            context, 
            overall_counts, 
            min_messages=5
        )
        
        self.assertIsNone(result)
    
    def test_check_for_fallback_below_threshold(self):
        """Test fallback check below negativity threshold"""
        context = [
            {'sentiment': 'positive'} for _ in range(5)
        ] + [
            {'sentiment': 'negative'} for _ in range(2)
        ]
        
        overall_counts = Counter({'positive': 5, 'negative': 2})
        
        # Should not trigger fallback (2/7 = 0.29 < 0.7)
        result = check_for_fallback(
            context,
            overall_counts,
            threshold=0.7,
            min_messages=5
        )
        
        self.assertIsNone(result)
    
    def test_check_for_fallback_empty_context(self):
        """Test fallback check with empty context"""
        result = check_for_fallback([], None, min_messages=5)
        self.assertIsNone(result)
    
    def test_check_for_fallback_none_context(self):
        """Test fallback check with None context"""
        result = check_for_fallback(None, None, min_messages=5)
        self.assertIsNone(result) 