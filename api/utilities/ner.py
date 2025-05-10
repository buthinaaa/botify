from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load tokenizer and model locally
ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER-uncased")
ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER-uncased")

# Create NER pipeline
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")

import requests
import time

def call_ner_api(text, max_retries=5, backoff_factor=1.0):
    """
    Runs NER using the locally loaded model (no API calls).

    Parameters:
    text (str): Input text for named entity recognition.

    Returns:
    list: List of extracted entities with 'entity_group', 'word', 'score', etc.
    """
    try:
        return ner_pipeline(text)
    except Exception as e:
        print(f"[NER Error] Local NER processing failed: {e}")
        return []

def extract_entities_from_ner_output(ner_output):
    """
    Parses the API response and returns a list of useful entity data.
    Each item: {'word': ..., 'entity': ..., 'score': ...}
    """
    return [
        {"word": ent["word"], "entity": ent["entity_group"], "score": ent["score"]}
        for ent in ner_output
    ]

def filter_high_confidence_entities(entities, min_score=0.7):
    """
    Keeps only entities with confidence >= min_score.
    """
    return [ent for ent in entities if ent["score"] >= min_score]

def update_context_with_ner(context, preprocessed_message, ner_entities):
    """
    Adds current message and its NER entities to the shared context.
    """
    context.append({
        "user": preprocessed_message["original"],
        "entities": ner_entities
    })

def extract_ner_entities(preprocessed_message, context):
    """
    Complete NER pipeline:
    - Uses clean_text from preprocessing
    - Calls API
    - Filters relevant entities
    - Prioritizes high-confidence ones
    - Updates context
    """
    message_data = preprocessed_message  # preprocessed_message already contains the result
    text = message_data["text_use"]  # Access the "text_use" key from the result
    ner_output = call_ner_api(text)
    extracted = extract_entities_from_ner_output(ner_output)
    prioritized = filter_high_confidence_entities(extracted)
    update_context_with_ner(context, preprocessed_message, prioritized)
    return prioritized
