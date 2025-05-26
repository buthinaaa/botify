from api.services.nlp_manager import NLPManager
import time
import requests

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
    try:
        nlp_manager = NLPManager.get_instance()
        result = nlp_manager.zero_shot_classifier(message, candidate_labels, multi_label=multi_label)
        labels = result.get("labels", [])
        scores = result.get("scores", [])
        return sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
    except Exception as e:
        print(f"[Intent Error] Intent recognition failed: {e}")
        return []
    
def classify_question_or_not(message, threshold=0.4):
    """
    Classifies if the message is a question, feedback, greeting, complaint, or out-of-scope.

    Parameters:
        message (str): The user's message.
        threshold (float): The minimum confidence score required to consider the intent valid.

    Returns:
        - is_question (bool): Whether the message is a question or not.
        - matched_labels (list): The list of labels above the threshold.
        - predictions (list): The predictions from the intent recognition API.
    """
    # Define the general intent labels to classify messages
    general_labels = ["support-question", "feedback", "complaint", "greeting","chitchat"]

    # Combine document-specific labels and general labels
    all_labels = general_labels

    # Get the predictions from the intent recognition API
    predictions = intent_recognition_api(message, all_labels, multi_label=False)

    matched_labels = max(predictions, key=lambda x: x[1])[0]
    # Check if the message is a question based on the labels
    is_question = "support-question" in matched_labels

    # Return results
    return is_question, matched_labels, predictions

def classify_topic_intent(message, doc_labels, threshold=0.3):
    """
    Classifies the message into specific business topic labels if it's a question.

    Parameters:
        message (str): The user's message (question).
        doc_labels (List[str]): The labels generated from the documents (topics).
        threshold (float): The minimum confidence score required to classify the topic.

    Returns:
        - matched_labels (list): The list of labels above the threshold.
        - predictions (list): The predictions from the intent recognition API for topic classification.
    """
    # Get the predictions from the intent recognition API using document-specific labels
    predictions = intent_recognition_api(message, doc_labels, multi_label=True)

    # Filter the matched labels based on the threshold
    matched_labels = [label for label, score in predictions if score >= threshold]

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
    if isinstance(matched_labels, str):
        matched_labels = [matched_labels]
    
    # Handle out-of-scope or casual chitchat
    if any(label in matched_labels for label in ["out-of-scope", "chitchat"]):
        return {
            "status": "fallback",
            "message": "I'm here to help with support-related questions. Please let me know how I can assist you 😊",
            "reason": "out-of-scope-or-chitchat"
        }

    # Everything looks good, proceed
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
    # Step 1: Classify general intent
    is_question, matched_labels, predictions = classify_question_or_not(message, threshold)

    # Step 2: Initialize result structure
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

    # Step 3: Early check for out-of-scope
    if any("out-of-scope" in label for label in matched_labels):
        result['safety_flags']['out_of_scope'] = True

    # Step 4: Classify topic intent if it's a question
    if is_question:
        topic_labels, _ = classify_topic_intent(message, doc_labels, threshold)
        result['topic_labels'] = topic_labels

        # No topic matched = needs clarification
        if not topic_labels:
            result['safety_flags']['needs_clarification'] = True
    else:
        # For non-question messages, assign topic labels as general intent
        result['topic_labels'] = matched_labels

    # Step 5: Run fallback logic
    fallback = handle_fallbacks(
        is_question=result['is_question'],
        matched_labels=result['matched_labels'],
        topic_labels=result['topic_labels']
    )
    result['fallback'] = fallback

    return result

