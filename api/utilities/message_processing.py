import re
import string
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker


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
    # This regex finds 3 or more of the same character and replaces with 2
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def preprocess_message(message, context=None):
    """
    Preprocess user messages for NLP tasks

    Parameters:
    message (str): The input message to preprocess
    context (list, optional): Message history context to update

    Returns:
    dict: Processed message components
    """
    # 1. Store original message for reference
    original = message

    # 2. Convert to lowercase
    text = message.lower()

    # 3. Handle punctuation (keep ! ? . but REMOVE emojis for text_use)
    punctuation_to_keep = "!?."
    filtered_chars = []
    for char in text:
        if char.isalnum() or char.isspace() or char in punctuation_to_keep:
            filtered_chars.append(char)
    text_no_emojis = ''.join(filtered_chars)  # used for text_use

    # 4. Normalize repeated characters
    text_no_emojis = normalize_repeated_chars(text_no_emojis)

    # 5. Tokenize for spell checking
    raw_tokens_no_emojis = text_no_emojis.split()

    # 6. Apply autocorrection with safeguards
    spell = SpellChecker()
    corrected_tokens_no_emojis = []
    for token in raw_tokens_no_emojis:
        if (len(token) <= 2 or
            not token.isalpha() or
            token in ["im", "its", "youre"]):
            corrected_tokens_no_emojis.append(token)
        else:
            correction = spell.correction(token)
            if correction and (correction == token or len(correction) > len(token) * 0.7):
                corrected_tokens_no_emojis.append(correction)
            else:
                corrected_tokens_no_emojis.append(token)

    # 7. Build text_use (no emojis, before stopwords/lemmatization)
    text_use = ' '.join(corrected_tokens_no_emojis)

    # 8. Detect emojis (from original message)
    emojis_used = [char for char in message if char in emoji.EMOJI_DATA]

    # 9. Now go back to full lowercased text with emojis for rest of pipeline
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
    stop_words = set(stopwords.words('english')) - {"not", "no", "very", "too"}
    tokens = [word for word in corrected_tokens if word not in stop_words]

    # 11. Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for token in tokens:
        lemma_v = lemmatizer.lemmatize(token, pos='v')
        lemma_n = lemmatizer.lemmatize(token, pos='n')
        lemmatized_tokens.append(lemma_v if len(lemma_v) < len(lemma_n) else lemma_n)

    # 12. Update context
    if context is not None:
        context.append({"user": original})

    # 13. Final clean text (used in retrieval, etc.)
    clean_text = ' '.join(tokens)

    return {
        "original": original,
        "clean_text": clean_text,
        "text_use": text_use,  # ✅ clean version without emojis
        "tokens": tokens,
        "lemmatized_tokens": lemmatized_tokens,
        "corrected_tokens": corrected_tokens,
        "emojis": emojis_used,
    }
