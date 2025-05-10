import requests
import time
# For processing sentiment API response
from collections import Counter

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load locally once
tokenizer_sentiment = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model.eval()  # Set model to evaluation mode

import time
import requests

def get_sentiment_from_api(text, max_retries=5, backoff_factor=1.0):
    """
    Uses the locally loaded sentiment model to classify the input text.

    Parameters:
    text (str): The input text.

    Returns:
    tuple: (label, score)
    """
    try:
        # Tokenize and predict
        inputs = tokenizer_sentiment(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)

        score, predicted_class = torch.max(probs, dim=1)
        label_map = ['negative', 'neutral', 'positive']  # Model class order

        return label_map[predicted_class.item()], score.item()

    except Exception as e:
        print(f"[Error] Sentiment prediction failed: {e}")
        return None, None
def analyze_sentiment(preprocessed_message, context=None, score_threshold=0.5, fallback_threshold=0.3):
    """
    Analyzes sentiment using the original text.
    If confidence is low, retries using recent context.
    Falls back to 'neutral' if confidence remains too low.
    """
    text = preprocessed_message["original"]
    label, score = get_sentiment_from_api(text)

    # Step 1: Low confidence? Use recent context if available
    if score < score_threshold and context:
        recent_texts = [entry["original"] for entry in context[-2:]]
        combined_text = " ".join(recent_texts + [text])
        label, score = get_sentiment_from_api(combined_text)

    # Step 2: Still low confidence? Fallback to neutral
    if score < fallback_threshold:
        label = "neutral"

    return label, score

def update_context_with_sentiment(context, preprocessed_message, sentiment_label):
    """
    Adds a new entry to the context with the sentiment label.
    """
    context.append({
        "user": preprocessed_message["original"],
        "sentiment": sentiment_label
    })

def get_overall_sentiment(context, window=10):
    """
    Looks at the last `window` messages and returns overall mood.
    """
    recent = context[-window:]
    sentiments = [entry.get("sentiment") for entry in recent if "sentiment" in entry]
    counts = Counter(sentiments)
    return counts

def check_for_fallback(context, overall_counts, window=10, min_messages=5, threshold=0.7, trend_window=5, trend_threshold=0.6):
    """
    Trigger fallback only if:
    - Total negative sentiment in last `window` exceeds `threshold`
    - AND most recent `trend_window` are also mostly negative (indicating no improvement)
    """
    total = sum(overall_counts.values())
    if total < min_messages:
        return  # Not enough data

    # Overall negativity
    negative_count = overall_counts.get("negative", 0)
    overall_ratio = negative_count / total

    if overall_ratio < threshold:
        return  # Not bad enough overall

    # Check recent trend
    recent_trend = context[-trend_window:]
    recent_sentiments = [msg.get("sentiment") for msg in recent_trend if "sentiment" in msg]
    if not recent_sentiments:
        return  # No valid sentiments

    negative_trend_count = recent_sentiments.count("negative")
    trend_ratio = negative_trend_count / len(recent_sentiments)

    if trend_ratio >= trend_threshold:
        print("🚨 Hey, we need a human here! 🚨")
    else:
        print("🟢 User sentiment is improving, no fallback needed.")

def sentiment_pipeline(message, context=None):
    """
    A simple pipeline function that wraps the 4 sentiment analysis functions.

    Parameters:
    message (str): The raw message text to analyze
    context (list): Previous context. If None, creates a new empty context

    Returns:
    dict: Result containing analysis data and updated context
    """
    # Initialize context if None
    if context is None:
        context = []

    # Step 1: Preprocess the message (simple version)
    preprocessed_message = {"original": message}

    # Step 2: Analyze sentiment
    sentiment_label, confidence = analyze_sentiment(
        preprocessed_message,
        context
    )

    # Step 3: Update context
    context = update_context_with_sentiment(context, preprocessed_message, sentiment_label)

    # Step 4: Get overall sentiment
    overall_sentiment = get_overall_sentiment(context)

    # Step 5: Check if fallback is needed - use the existing function directly
    check_for_fallback(
        context,
        overall_sentiment
    )

    return {
        "message": message,
        "sentiment": sentiment_label,
        "confidence": confidence,
        "context": context,  # Return updated context
        "overall_sentiment": dict(overall_sentiment)
    }

