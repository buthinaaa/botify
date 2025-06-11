import logging
import time
import traceback
from api.services.nlp_manager import NLPManager
import time
import requests

# Configure logger
logger = logging.getLogger(__name__)

def intent_recognition_api(message, candidate_labels, multi_label=True, max_retries=5, backoff_factor=1.0):
    """
    Uses centralized model from NLPManager for zero-shot intent recognition.

    Parameters:
        message (str): The user's message.
        candidate_labels (List[str]): The list of possible intent labels.
        multi_label (bool): Whether to allow multiple labels.

    Returns:
        List of tuples: [(label, score), ...] sorted by confidence.
    """
    logger.warning("intent_recognition_api started")
    logger.debug(f"Message: '{message[:100]}...'")
    logger.debug(f"Candidate labels: {candidate_labels}")
    logger.debug(f"Multi-label: {multi_label}")
    
    start_time = time.time()
    
    try:
        nlp_manager = NLPManager.get_instance()
        logger.warning("Retrieved NLPManager instance")
        
        result = nlp_manager.zero_shot_classifier(message, candidate_labels, multi_label=multi_label)
        logger.debug(f"Raw classifier result: {result}")
        
        labels = result.get("labels", [])
        scores = result.get("scores", [])
        sorted_results = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
        
        end_time = time.time()
        logger.warning(f"intent_recognition_api completed in {end_time - start_time:.3f}s")
        logger.warning(f"Results: {[(label, f'{score:.3f}') for label, score in sorted_results[:3]]}")
        
        return sorted_results
        
    except Exception as e:
        end_time = time.time()
        logger.error(f"intent_recognition_api failed after {end_time - start_time:.3f}s: {str(e)}")
        logger.error(traceback.format_exc())
        return []
    
def classify_question_or_not(message, threshold=0.4):
    """
    Classifies if the message is a question, feedback, greeting, complaint, or out-of-scope.
    """
    logger.warning("classify_question_or_not started")
    logger.debug(f"Message: '{message[:100]}...'")
    logger.debug(f"Threshold: {threshold}")
    
    start_time = time.time()
    
    # Define the general intent labels to classify messages
    general_labels = ["support-question","support", "complaint", "appreciation", "greeting", "chitchat"]
    logger.warning(f"Using general labels: {general_labels}")
    
    # Combine document-specific labels and general labels
    all_labels = general_labels

    # Get the predictions from the intent recognition API
    logger.warning("Getting predictions from intent recognition API")
    predictions = intent_recognition_api(message, all_labels, multi_label=False)
    logger.debug(f"API predictions: {predictions}")

    if not predictions:
        logger.warning("No predictions returned from API")
        end_time = time.time()
        logger.warning(f"classify_question_or_not completed (no predictions) in {end_time - start_time:.3f}s")
        return False, [], []

    # Filter labels based on the threshold
    matched_labels = max(predictions, key=lambda x: x[1])[0]
    logger.warning(f"Matched label: {matched_labels}")

    # Check if the message is a question based on the labels
    is_question = "support-question" in matched_labels or "support" in matched_labels
    logger.warning(f"Is question: {is_question}")
    
    end_time = time.time()
    logger.warning(f"classify_question_or_not completed in {end_time - start_time:.3f}s")
    
    # Return results
    return is_question, matched_labels, predictions


def classify_topic_intent(message, doc_labels, threshold=0.3):
    """
    Classifies the message into specific business topic labels if it's a question.
    """
    logger.warning("classify_topic_intent started")
    logger.debug(f"Message: '{message[:100]}...'")
    logger.debug(f"Document labels: {doc_labels}")
    logger.debug(f"Threshold: {threshold}")
    
    start_time = time.time()

    if not doc_labels:
        logger.warning("No document labels provided")
        end_time = time.time()
        logger.warning(f"classify_topic_intent completed (no labels) in {end_time - start_time:.3f}s")
        return [], []

    # Get the predictions from the intent recognition API using document-specific labels
    logger.warning("Getting topic predictions from intent recognition API")
    predictions = intent_recognition_api(message, doc_labels, multi_label=True)
    logger.debug(f"Topic predictions: {predictions}")

    # Filter the matched labels based on the threshold
    matched_labels = [label for label, score in predictions if score >= threshold]
    logger.warning(f"Matched labels above threshold {threshold}: {matched_labels}")

    end_time = time.time()
    logger.warning(f"classify_topic_intent completed in {end_time - start_time:.3f}s")

    # Return the matched labels and predictions
    return matched_labels, predictions

def handle_fallbacks(is_question, matched_labels, topic_labels):
    """
    Handles fallback responses based on detected intent and topic classification.

    Parameters:
        is_question (bool): Whether the message was identified as a question.
        matched_labels (list): General intent labels matched (e.g., "question", "feedback").
        topic_labels (list): Business-specific topic labels matched.

    Returns:
        dict: {
            "status": "fallback" or "ok",
            "message": str or None (message to return to user if fallback),
            "reason": str or None (why fallback was triggered)
        }
    """
    logger.warning("handle_fallbacks started")
    logger.debug(f"Is question: {is_question}")
    logger.debug(f"Matched labels: {matched_labels}")
    logger.debug(f"Topic labels: {topic_labels}")
    
    start_time = time.time()
    
    if isinstance(matched_labels, str):
        matched_labels = [matched_labels]
        logger.debug(f"Converted matched_labels to list: {matched_labels}")
    
    # Handle out-of-scope or casual chitchat
    out_of_scope_labels = ["out-of-scope", "chitchat"]
    found_fallback_labels = [label for label in out_of_scope_labels if any(label in ml for ml in matched_labels)]
    
    if found_fallback_labels:
        logger.warning(f"Fallback triggered by labels: {found_fallback_labels}")
        end_time = time.time()
        logger.warning(f"handle_fallbacks completed (fallback) in {end_time - start_time:.3f}s")
        
        return {
            "status": "fallback",
            "message": "I'm here to help with support-related questions. Please let me know how I can assist you 😊",
            "reason": "out-of-scope-or-chitchat"
        }

    # Everything looks good, proceed
    logger.warning("No fallback needed, proceeding normally")
    end_time = time.time()
    logger.warning(f"handle_fallbacks completed (ok) in {end_time - start_time:.3f}s")
    
    return {
        "status": "ok",
        "message": None,
        "reason": None
    }

    # Handle out-of-scope or casual chitchat
    # if "out-of-scope" in matched_labels or "chitchat" in matched_labels:
    #     return {
    #         "status": "fallback",
    #         "message": "I'm here to help with support-related questions. Please let me know how I can assist you 😊",
    #         "reason": "out-of-scope-or-chitchat"
    #     }


    # # Everything looks good, proceed
    # return {
    #     "status": "ok",
    #     "message": None,
    #     "reason": None
    # }

def detect_intent(message, doc_labels, threshold=0.4, max_fallback=3):
    """
    Main function to classify and return the intent of the user's message,
    including specific labels, and handle fallback logic if necessary.

    Parameters:
        message (str): The user's message.
        doc_labels (List[str]): The labels generated from the documents (topics).
        threshold (float): The minimum confidence score required to classify an intent.
        max_fallback (int): The maximum number of clarifications before asking for human support.

    Returns:
        dict: A dictionary with keys such as:
            - 'is_question'
            - 'matched_labels'
            - 'topic_labels'
            - 'safety_flags'
            - 'fallback': {
                  'status': "ok" or "fallback",
                  'message': fallback text or None,
                  'reason': why fallback was triggered
              }
    """
    logger.warning("detect_intent started")
    logger.debug(f"Message: '{message[:100]}...'")
    logger.debug(f"Document labels: {doc_labels}")
    logger.debug(f"Threshold: {threshold}")
    logger.debug(f"Max fallback: {max_fallback}")
    
    start_time = time.time()

    # Step 1: Classify general intent
    logger.warning("STEP 1: Classifying general intent")
    general_start = time.time()
    is_question, matched_labels, predictions = classify_question_or_not(message, threshold)
    general_end = time.time()
    logger.warning(f"General intent classification completed in {general_end - general_start:.3f}s")

    # Step 2: Initialize result structure
    logger.warning("STEP 2: Initializing result structure")
    result = {
        'is_question': is_question,
        'matched_labels': matched_labels,
        'topic_labels': [],
        'safety_flags': {
            'needs_clarification': False,
            'out_of_scope': False
        },
        'fallback': {
            'status': 'ok',
            'message': None,
            'reason': None
        }
    }
    logger.debug(f"Initial result structure: {result}")

    # Step 3: Early check for out-of-scope
    logger.warning("STEP 3: Checking for out-of-scope")
    out_of_scope_found = any("out-of-scope" in str(label) for label in [matched_labels] if matched_labels)
    if out_of_scope_found:
        result['safety_flags']['out_of_scope'] = True
        logger.warning("Out-of-scope detected")

    # Step 4: Classify topic intent if it's a question
    logger.warning("STEP 4: Topic intent classification")
    if is_question:
        logger.warning("Question detected, performing topic classification")
        topic_start = time.time()
        topic_labels, _ = classify_topic_intent(message, doc_labels, threshold)
        topic_end = time.time()
        logger.warning(f"Topic classification completed in {topic_end - topic_start:.3f}s")
        
        result['topic_labels'] = topic_labels
        logger.warning(f"Topic labels found: {topic_labels}")

        # No topic matched = needs clarification
        if not topic_labels:
            result['safety_flags']['needs_clarification'] = True
            logger.warning("No topic labels matched, needs clarification")
    else:
        # For non-question messages, assign topic labels as general intent
        result['topic_labels'] = [matched_labels] if isinstance(matched_labels, str) else matched_labels
        logger.warning(f"Non-question: using general intent as topic labels: {result['topic_labels']}")

    # Step 5: Run fallback logic
    logger.warning("STEP 5: Running fallback logic")
    fallback_start = time.time()
    fallback = handle_fallbacks(
        is_question=result['is_question'],
        matched_labels=result['matched_labels'],
        topic_labels=result['topic_labels']
    )
    fallback_end = time.time()
    logger.warning(f"Fallback logic completed in {fallback_end - fallback_start:.3f}s")
    
    result['fallback'] = fallback
    logger.warning(f"Fallback result: {fallback}")

    end_time = time.time()
    total_time = end_time - start_time
    logger.warning(f"detect_intent completed in {total_time:.3f}s")
    logger.warning(f"Final result: {result}")

    return result

