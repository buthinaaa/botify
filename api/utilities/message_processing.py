import re
import string
import emoji
import nltk
import logging
import time
import traceback
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

# Configure logger
logger = logging.getLogger(__name__)

# Ensure NLTK resources are downloaded
try:
    stopwords.words('english')
    WordNetLemmatizer().lemmatize('test')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

def normalize_repeated_chars(text):
    """
    Normalize words with repeated characters (e.g., 'sooooo' -> 'so')
    Only keeps up to 2 consecutive occurrences of the same character
    """
    logger.debug("normalize_repeated_chars started")
    logger.debug(f"Input text: '{text[:100]}...'")
    
    # This regex finds 3 or more of the same character and replaces with 2
    result = re.sub(r'(.)\1{2,}', r'\1\1', text)
    logger.debug(f"Normalized text: '{result[:100]}...'")
    return result

def preprocess_message(message, context=None):
    """
    Preprocess user messages for NLP tasks

    Parameters:
    message (str): The input message to preprocess
    context (list, optional): Message history context to update

    Returns:
    dict: Processed message components
    """
    logger.warning("preprocess_message started")
    logger.debug(f"Input message: '{message[:100]}...'")
    logger.debug(f"Context provided: {'Yes' if context else 'No'}")
    
    start_time = time.time()
    
    try:
        # 1. Store original message for reference
        original = message
        logger.debug("Stored original message")

        # 2. Convert to lowercase
        text = message.lower()
        logger.debug("Converted to lowercase")

        # 3. Handle punctuation (keep ! ? . but REMOVE emojis for text_use)
        logger.warning("Processing punctuation and filtering characters")
        punctuation_to_keep = "!?."
        filtered_chars = []
        for char in text:
            if char.isalnum() or char.isspace() or char in punctuation_to_keep:
                filtered_chars.append(char)
        text_no_emojis = ''.join(filtered_chars)  # used for text_use
        logger.debug(f"Text without emojis: '{text_no_emojis[:100]}...'")

        # 4. Normalize repeated characters
        logger.warning("Normalizing repeated characters")
        text_no_emojis = normalize_repeated_chars(text_no_emojis)

        # 5. Tokenize for spell checking
        logger.warning("Tokenizing for spell checking")
        raw_tokens_no_emojis = text_no_emojis.split()
        logger.debug(f"Raw tokens count: {len(raw_tokens_no_emojis)}")

        # 6. Apply autocorrection with safeguards
        logger.warning("Applying spell correction")
        spell_start = time.time()
        spell = SpellChecker()
        corrected_tokens_no_emojis = []
        corrections_made = 0
        
        for token in raw_tokens_no_emojis:
            if (len(token) <= 2 or
                not token.isalpha() or
                token in ["im", "its", "youre"]):
                corrected_tokens_no_emojis.append(token)
            else:
                correction = spell.correction(token)
                if correction and (correction == token or len(correction) > len(token) * 0.7):
                    if correction != token:
                        corrections_made += 1
                        logger.debug(f"Corrected '{token}' -> '{correction}'")
                    corrected_tokens_no_emojis.append(correction)
                else:
                    corrected_tokens_no_emojis.append(token)
        
        spell_end = time.time()
        logger.warning(f"Spell correction completed in {spell_end - spell_start:.3f}s, made {corrections_made} corrections")

        # 7. Build text_use (no emojis, before stopwords/lemmatization)
        text_use = ' '.join(corrected_tokens_no_emojis)
        logger.debug(f"Text_use: '{text_use[:100]}...'")

        # 8. Detect emojis (from original message)
        logger.warning("Detecting emojis")
        emojis_used = [char for char in message if char in emoji.EMOJI_DATA]
        logger.debug(f"Emojis found: {emojis_used}")

        # 9. Now go back to full lowercased text with emojis for rest of pipeline
        logger.warning("Processing text with emojis for full pipeline")
        # Filter again, this time allowing emojis
        text_with_emojis = []
        for char in text:
            if char in emoji.EMOJI_DATA or char.isalnum() or char.isspace() or char in punctuation_to_keep:
                text_with_emojis.append(char)
        text_with_emojis = ''.join(text_with_emojis)
        text_with_emojis = normalize_repeated_chars(text_with_emojis)

        raw_tokens = text_with_emojis.split()

        corrected_tokens = []
        for token in raw_tokens:
            if (token in emoji.EMOJI_DATA or
                len(token) <= 2 or
                not token.isalpha() or
                token in ["im", "its", "youre"]):
                corrected_tokens.append(token)
            else:
                correction = spell.correction(token)
                if correction and (correction == token or len(correction) > len(token) * 0.7):
                    corrected_tokens.append(correction)
                else:
                    corrected_tokens.append(token)

        # 10. Remove stopwords
        logger.warning("Removing stopwords")
        stop_words = set(stopwords.words('english')) - {"not", "no", "very", "too"}
        tokens = [word for word in corrected_tokens if word not in stop_words]
        logger.debug(f"Tokens after stopword removal: {len(tokens)} (from {len(corrected_tokens)})")

        # 11. Lemmatize
        logger.warning("Lemmatizing tokens")
        lemma_start = time.time()
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = []
        for token in tokens:
            lemma_v = lemmatizer.lemmatize(token, pos='v')
            lemma_n = lemmatizer.lemmatize(token, pos='n')
            lemmatized_tokens.append(lemma_v if len(lemma_v) < len(lemma_n) else lemma_n)
        lemma_end = time.time()
        logger.warning(f"Lemmatization completed in {lemma_end - lemma_start:.3f}s")

        # 12. Update context
        if context is not None:
            context.append({"user": original})
            logger.debug("Updated context with original message")

        # 13. Final clean text (used in retrieval, etc.)
        clean_text = ' '.join(tokens)
        logger.debug(f"Clean text: '{clean_text[:100]}...'")

        result = {
            "original": original,
            "clean_text": clean_text,
            "text_use": text_use,  # clean version without emojis
            "tokens": tokens,
            "lemmatized_tokens": lemmatized_tokens,
            "corrected_tokens": corrected_tokens,
            "emojis": emojis_used,
        }

        end_time = time.time()
        logger.warning(f"preprocess_message completed in {end_time - start_time:.3f}s")
        logger.debug(f"Processing result summary:")
        logger.debug(f"- Original length: {len(original)}")
        logger.debug(f"- Clean text length: {len(clean_text)}")
        logger.debug(f"- Text_use length: {len(text_use)}")
        logger.debug(f"- Tokens count: {len(tokens)}")
        logger.debug(f"- Emojis count: {len(emojis_used)}")

        return result
        
    except Exception as e:
        end_time = time.time()
        logger.error(f"preprocess_message failed after {end_time - start_time:.3f}s: {str(e)}")
        logger.error(traceback.format_exc())
        # Return minimal result structure on error
        return {
            "original": message,
            "clean_text": message.lower(),
            "text_use": message.lower(),
            "tokens": message.lower().split(),
            "lemmatized_tokens": message.lower().split(),
            "corrected_tokens": message.lower().split(),
            "emojis": [],
        }
