import logging
import time
import traceback
from api.services.nlp_manager import NLPManager

# Configure logger
logger = logging.getLogger(__name__)

def call_ner_api(text, max_retries=5, backoff_factor=1.0):
    """
    Runs NER using the centralized model from NLPManager.

    Parameters:
    text (str): Input text for named entity recognition.

    Returns:
    list: List of extracted entities with 'entity_group', 'word', 'score', etc.
    """
    logger.warning("call_ner_api started")
    logger.debug(f"Input text: '{text[:100]}...'")
    
    start_time = time.time()
    
    try:
        nlp_manager = NLPManager.get_instance()
        logger.warning("Retrieved NLPManager instance")
        
        result = nlp_manager.ner_pipeline(text)
        logger.debug(f"Raw NER result: {result}")
        
        end_time = time.time()
        logger.warning(f"call_ner_api completed in {end_time - start_time:.3f}s")
        logger.warning(f"Extracted {len(result)} raw entities")
        
        return result
        
    except Exception as e:
        end_time = time.time()
        logger.error(f"call_ner_api failed after {end_time - start_time:.3f}s: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def extract_entities_from_ner_output(ner_output):
    """
    Parses the API response and returns a list of useful entity data.
    Each item: {'word': ..., 'entity': ..., 'score': ...}
    """
    logger.warning("extract_entities_from_ner_output started")
    logger.debug(f"Input NER output count: {len(ner_output) if ner_output else 0}")
    
    start_time = time.time()
    
    if not ner_output:
        logger.warning("No NER output provided")
        return []
    
    result = []
    for i, ent in enumerate(ner_output):
        try:
            entity_data = {
                "word": ent["word"], 
                "entity": ent["entity_group"], 
                "score": ent["score"]
            }
            result.append(entity_data)
            logger.debug(f"Entity {i+1}: {entity_data}")
        except KeyError as e:
            logger.warning(f"Skipping entity {i+1} due to missing key: {e}")
            continue
    
    end_time = time.time()
    logger.warning(f"extract_entities_from_ner_output completed in {end_time - start_time:.3f}s")
    logger.warning(f"Extracted {len(result)} valid entities from {len(ner_output)} raw entities")
    
    return result

def filter_high_confidence_entities(entities, min_score=0.7):
    """
    Keeps only entities with confidence >= min_score.
    """
    logger.warning("filter_high_confidence_entities started")
    logger.debug(f"Input entities count: {len(entities)}")
    logger.debug(f"Minimum score threshold: {min_score}")
    
    start_time = time.time()
    
    if not entities:
        logger.warning("No entities provided for filtering")
        return []
    
    filtered_entities = []
    for i, ent in enumerate(entities):
        if ent["score"] >= min_score:
            filtered_entities.append(ent)
            logger.debug(f"Kept entity {i+1}: {ent['word']} ({ent['entity']}) - score: {ent['score']:.3f}")
        else:
            logger.debug(f"Filtered out entity {i+1}: {ent['word']} ({ent['entity']}) - score: {ent['score']:.3f} < {min_score}")
    
    end_time = time.time()
    logger.warning(f"filter_high_confidence_entities completed in {end_time - start_time:.3f}s")
    logger.warning(f"Kept {len(filtered_entities)} out of {len(entities)} entities")
    
    return filtered_entities

def update_context_with_ner(context, preprocessed_message, ner_entities):
    """
    Adds current message and its NER entities to the shared context.
    """
    logger.warning("update_context_with_ner started")
    logger.debug(f"Context length before: {len(context) if context else 0}")
    logger.debug(f"NER entities count: {len(ner_entities)}")
    
    start_time = time.time()
    
    context.append({
        "user": preprocessed_message["original"],
        "entities": ner_entities
    })
    
    end_time = time.time()
    logger.warning(f"update_context_with_ner completed in {end_time - start_time:.3f}s")
    logger.debug(f"Context length after: {len(context)}")

def extract_ner_entities(preprocessed_message, context):
    """
    Complete NER pipeline:
    - Uses clean_text from preprocessing
    - Calls API
    - Filters relevant entities
    - Prioritizes high-confidence ones
    - Updates context
    """
    logger.warning("extract_ner_entities started")
    logger.debug(f"Preprocessed message keys: {list(preprocessed_message.keys())}")
    
    start_time = time.time()
    
    try:
        message_data = preprocessed_message  # preprocessed_message already contains the result
        text = message_data["text_use"]  # Access the "text_use" key from the result
        logger.warning(f"Using text_use for NER: '{text[:100]}...'")
        
        # Step 1: Call NER API
        logger.warning("STEP 1: Calling NER API")
        api_start = time.time()
        ner_output = call_ner_api(text)
        api_end = time.time()
        logger.warning(f"NER API call completed in {api_end - api_start:.3f}s")
        
        # Step 2: Extract entities from output
        logger.warning("STEP 2: Extracting entities from output")
        extract_start = time.time()
        extracted = extract_entities_from_ner_output(ner_output)
        extract_end = time.time()
        logger.warning(f"Entity extraction completed in {extract_end - extract_start:.3f}s")
        
        # Step 3: Filter high-confidence entities
        logger.warning("STEP 3: Filtering high-confidence entities")
        filter_start = time.time()
        prioritized = filter_high_confidence_entities(extracted)
        filter_end = time.time()
        logger.warning(f"Entity filtering completed in {filter_end - filter_start:.3f}s")
        
        # Step 4: Update context
        logger.warning("STEP 4: Updating context")
        context_start = time.time()
        update_context_with_ner(context, preprocessed_message, prioritized)
        context_end = time.time()
        logger.warning(f"Context update completed in {context_end - context_start:.3f}s")
        
        end_time = time.time()
        total_time = end_time - start_time
        logger.warning(f"extract_ner_entities completed in {total_time:.3f}s")
        logger.warning(f"Final result: {len(prioritized)} high-confidence entities")
        
        # Log entity summary
        if prioritized:
            entity_summary = [(ent['word'], ent['entity'], f"{ent['score']:.3f}") for ent in prioritized]
            logger.warning(f"Extracted entities: {entity_summary}")
        else:
            logger.warning("No high-confidence entities extracted")
        
        return prioritized
        
    except Exception as e:
        end_time = time.time()
        logger.error(f"extract_ner_entities failed after {end_time - start_time:.3f}s: {str(e)}")
        logger.error(traceback.format_exc())
        # Return empty list on error
        return []
