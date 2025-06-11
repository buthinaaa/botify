import requests
import time
import logging
import traceback
# For processing sentiment API response
from collections import Counter
import torch
import torch.nn.functional as F
from api.services.nlp_manager import NLPManager

# Configure logger
logger = logging.getLogger(__name__)

def get_sentiment_from_api(text, max_retries=5, backoff_factor=1.0):
    """
    Uses the centralized sentiment model from NLPManager to classify the input text.
    """
    logger.warning("get_sentiment_from_api started")
    
    start_time = time.time()
    
    try:
        # Get model and tokenizer from NLPManager
        logger.warning("Getting model and tokenizer from NLPManager")
        nlp_manager = NLPManager.get_instance()
        tokenizer = nlp_manager.sentiment_tokenizer
        model = nlp_manager.sentiment_model
        logger.warning("Successfully retrieved sentiment model and tokenizer")
        
        # Tokenize and predict
        logger.warning("Tokenizing input text")
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        logger.debug(f"Tokenized input shape: {inputs.input_ids.shape}")
        
        logger.warning("Running sentiment prediction")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)

        score, predicted_class = torch.max(probs, dim=1)
        label_map = ['negative', 'neutral', 'positive']  # Model class order

        predicted_label = label_map[predicted_class.item()]
        confidence_score = score.item()
        
        end_time = time.time()
        logger.warning(f"get_sentiment_from_api completed in {end_time - start_time:.3f}s")
        logger.warning(f"Predicted sentiment: {predicted_label} (confidence: {confidence_score:.3f})")

        return predicted_label, confidence_score

    except Exception as e:
        end_time = time.time()
        logger.error(f"get_sentiment_from_api failed after {end_time - start_time:.3f}s: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def analyze_sentiment(preprocessed_message, context=None, score_threshold=0.5, fallback_threshold=0.3):
    """
    Analyzes sentiment using the original text.
    If confidence is low, retries using recent context.
    Falls back to 'neutral' if confidence remains too low.
    """
    logger.warning("analyze_sentiment started")
    logger.debug(f"Score threshold: {score_threshold}")
    logger.debug(f"Fallback threshold: {fallback_threshold}")
    logger.debug(f"Context provided: {'Yes' if context else 'No'}")
    
    start_time = time.time()
    
    text = preprocessed_message["original"]
    
    # Step 1: Initial sentiment analysis
    logger.warning("STEP 1: Initial sentiment analysis")
    initial_start = time.time()
    label, score = get_sentiment_from_api(text)
    initial_end = time.time()
    logger.warning(f"Initial analysis completed in {initial_end - initial_start:.3f}s")
    
    if label is None:
        logger.error("Initial sentiment analysis failed, returning neutral")
        return "neutral", 0.0

    logger.warning(f"Initial result: {label} (score: {score:.3f})")

    # Step 1: Low confidence? Use recent context if available
    if score < score_threshold and context:
        logger.warning(f"Low confidence ({score:.3f} < {score_threshold}), trying with context")
        context_start = time.time()
        
        recent_texts = [entry.get("original", "") for entry in context if "original" in entry]
        if recent_texts:
            combined_text = " ".join(recent_texts + [text])
            logger.warning(f"Combined text with context: '{combined_text[:100]}...'")
            
            label, score = get_sentiment_from_api(combined_text)
            context_end = time.time()
            logger.warning(f"Context-based analysis completed in {context_end - context_start:.3f}s")
            logger.warning(f"Context result: {label} (score: {score:.3f})")
        else:
            logger.warning("No valid context entries found for enhancement")

    # Step 2: Still low confidence? Fallback to neutral
    if score and score < fallback_threshold:
        logger.warning(f"Still low confidence ({score:.3f} < {fallback_threshold}), falling back to neutral")
        label = "neutral"

    end_time = time.time()
    logger.warning(f"analyze_sentiment completed in {end_time - start_time:.3f}s")
    logger.warning(f"Final sentiment: {label} (score: {score:.3f})")

    return label, score

def update_context_with_sentiment(context, preprocessed_message, sentiment_label):
    """
    Adds a new entry to the context with the sentiment label.
    """
    logger.warning("update_context_with_sentiment started")
    logger.debug(f"Sentiment label: {sentiment_label}")
    logger.debug(f"Context length before: {len(context) if context else 0}")
    
    start_time = time.time()
    
    context.append({
        "user": preprocessed_message["original"],
        "sentiment": sentiment_label
    })
    
    end_time = time.time()
    logger.warning(f"update_context_with_sentiment completed in {end_time - start_time:.3f}s")
    logger.debug(f"Context length after: {len(context)}")

def get_overall_sentiment(context, window=10):
    """
    Looks at the last `window` messages and returns overall mood.
    """
    logger.warning("get_overall_sentiment started")
    logger.debug(f"Window size: {window}")
    
    if context is None:
        context = []
        logger.warning("Context is None, using empty list")
    
    logger.debug(f"Total context entries: {len(context)}")
    
    start_time = time.time()
    print("received context in get_overall_sentiment", context)
    recent = context[-window:]
    sentiments = [entry.get("sentiment") for entry in recent if "sentiment" in entry]
    counts = Counter(sentiments)
    print("sentiments", sentiments)
    print("counts", counts)
    end_time = time.time()
    logger.warning(f"get_overall_sentiment completed in {end_time - start_time:.3f}s")
    logger.warning(f"Sentiment distribution in last {window} messages: {dict(counts)}")
    
    return counts

def check_for_fallback(context, overall_counts=None, window=10, min_messages=5, threshold=0.7, trend_window=5, trend_threshold=0.6) -> bool:
    """
    Trigger fallback only if:
    - Total negative sentiment in last `window` exceeds `threshold`
    - AND most recent `trend_window` are also mostly negative (indicating no improvement)
    """
    logger.warning("check_for_fallback started")
    logger.debug(f"Parameters: window={window}, min_messages={min_messages}, threshold={threshold}")
    logger.debug(f"trend_window={trend_window}, trend_threshold={trend_threshold}")
    
    if context is None:
        context = []
        logger.warning("Context is None, using empty list")
    
    start_time = time.time()
    
    if overall_counts:
        total = sum(overall_counts.values())
        logger.warning(f"Total messages analyzed: {total}")
        
        if total < min_messages:
            logger.warning(f"Not enough messages ({total} < {min_messages}), no fallback check")
            return  # Not enough data

        # Overall negativity
        negative_count = overall_counts.get("negative", 0)
        overall_ratio = negative_count / total
        logger.warning(f"Overall negative ratio: {overall_ratio:.3f} ({negative_count}/{total})")

        if overall_ratio < threshold:
            logger.warning(f"Overall negativity below threshold ({overall_ratio:.3f} < {threshold}), no fallback")
            return  # Not bad enough overall

    # Check recent trend
    logger.warning("Checking recent trend for improvement")
    recent_trend = context[-trend_window:]
    recent_sentiments = [msg.get("sentiment") for msg in recent_trend if "sentiment" in msg]
    logger.debug(f"Recent sentiments: {recent_sentiments}")
    
    if not recent_sentiments:
        logger.warning("No valid recent sentiments found")
        return  # No valid sentiments

    negative_trend_count = recent_sentiments.count("negative")
    trend_ratio = negative_trend_count / len(recent_sentiments)
    logger.warning(f"Recent trend negative ratio: {trend_ratio:.3f} ({negative_trend_count}/{len(recent_sentiments)})")

    end_time = time.time()
    logger.warning(f"check_for_fallback completed in {end_time - start_time:.3f}s")

    if trend_ratio >= trend_threshold:
        logger.warning("🚨 ESCALATION NEEDED: Persistent negative sentiment detected!")
        return True
    else:
        logger.warning("🟢 User sentiment is improving, no fallback needed")
        print("🟢 User sentiment is improving, no fallback needed.")
    
    return False

def sentiment_pipeline(message, context=None):
    """
    A simple pipeline function that wraps the 4 sentiment analysis functions.
    """
    logger.warning("sentiment_pipeline started")
    # logger.debug(f"Message: '{message[:100]}...'")
    logger.debug(f"Context provided: {'Yes' if context else 'No'}")
    
    start_time = time.time()

    # Initialize context if None
    if context is None:
        context = []
        logger.warning("Initialized empty context")

    # Step 1: Preprocess the message (simple version)
    logger.warning("STEP 1: Creating preprocessed message structure")
    preprocessed_message = {"original": message}

    # Step 2: Analyze sentiment
    logger.warning("STEP 2: Analyzing sentiment")
    sentiment_start = time.time()
    sentiment_label, confidence = analyze_sentiment(
        preprocessed_message,
        context
    )
    sentiment_end = time.time()
    logger.warning(f"Sentiment analysis completed in {sentiment_end - sentiment_start:.3f}s")

    # Step 3: Update context
    logger.warning("STEP 3: Updating context")
    context_start = time.time()
    print("context before update", context)
    update_context_with_sentiment(context, preprocessed_message, sentiment_label)
    print("context after update", context)
    context_end = time.time()
    logger.warning(f"Context update completed in {context_end - context_start:.3f}s")

    # Step 4: Get overall sentiment
    logger.warning("STEP 4: Getting overall sentiment")
    overall_start = time.time()
    overall_sentiment = get_overall_sentiment(context)
    overall_end = time.time()
    logger.warning(f"Overall sentiment analysis completed in {overall_end - overall_start:.3f}s")

    # Step 5: Check if fallback is needed - use the existing function directly
    logger.warning("STEP 5: Checking for fallback")
    fallback_start = time.time()
    check_for_fallback(
        context,
        overall_sentiment
    )
    fallback_end = time.time()
    logger.warning(f"Fallback check completed in {fallback_end - fallback_start:.3f}s")

    end_time = time.time()
    total_time = end_time - start_time
    logger.warning(f"sentiment_pipeline completed in {total_time:.3f}s")

    result = {
        "message": message,
        "sentiment": sentiment_label,
        "confidence": confidence,
        "context": context,  # Return updated context
        "overall_sentiment": dict(overall_sentiment)
    }
    
    logger.warning(f"Pipeline result: sentiment={sentiment_label}, confidence={confidence:.3f}")
    logger.debug(f"Overall sentiment distribution: {result['overall_sentiment']}")

    return result

