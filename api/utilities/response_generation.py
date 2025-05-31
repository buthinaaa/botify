
from api.utilities.intent_recognition import detect_intent
from api.utilities.message_processing import preprocess_message
from api.utilities.ner import extract_ner_entities
from api.utilities.retrieval_system import hybrid_search
from api.utilities.sentiment_analysis import analyze_sentiment, update_context_with_sentiment
from api.services.nlp_manager import NLPManager

def call_godel_api(prompt):
    nlp_manager = NLPManager.get_instance()
    nlp_manager.ensure_resources()
    
    model = nlp_manager.get_instance().response_model
    tokenizer = nlp_manager.get_instance().response_tokenizer
    try:
        # Ensure max length is within safe range
        model.model_max_length = 1024

        # Tokenize with truncation
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(model.device)

        # Generate response with tuned parameters
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,             # Give more space for response
            do_sample=True,                 # Enable sampling for varied responses
            temperature=0.7,                # Creativity control (lower = more deterministic)
            top_p=0.9,                      # Nucleus sampling for coherence
            top_k=50,                       # Optional: filter top-k tokens
            repetition_penalty=1.2,         # Penalize repetition (1.1–1.5)
            no_repeat_ngram_size=3,        # Avoid repeating phrases
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        # Decode and extract response
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        print("FULL RAW RESPONSE:", full_output)  # DEBUG
        print("Input length:", inputs.input_ids.shape[1])
        return full_output

    except Exception as e:
        print(f"[TinyLLaMA Error] {str(e)}")
        return "Error generating response."
    
def build_godel_prompt(user_message, sentiment, ner_entities, conversation_history, retrieved_chunks=None, intent_label=None, business_name="your business"):
    """
    Builds prompts compatible with decoder-only chat-style models like TinyLLaMA.
    Each output looks like a chat conversation with a User and Assistant.
    """
    # Normalize intent
    valid_intents = ["support-question", "support", "complaint", "appreciation", "greeting"]
    if isinstance(intent_label, list):
        primary_intent = next((i.lower() for i in intent_label if i.lower() in valid_intents), None)
    else:
        primary_intent = intent_label.lower() if intent_label and intent_label.lower() in valid_intents else None

    # Prepare context
    knowledge = prepare_knowledge(retrieved_chunks)
    entities = ", ".join(ner_entities) if ner_entities else None

    # Build chat-style prompt
    system_setup = f"You are a helpful and professional assistant at {business_name}.\n"

    user_prefix = "User:"
    assistant_prefix = "Assistant:"

    # Compose User turn based on intent
    if primary_intent == "support-question":
        user_block = f"{user_prefix} I need help with the following question. My sentiment is {sentiment}."
        if entities:
            user_block += f" This involves: {entities}."
        if retrieved_chunks:
            user_block += f"\nHere is some background info:\n{knowledge}"
        user_block += f"\n\n{user_message.strip()}"

    elif primary_intent == "support":
        user_block = f"{user_prefix} I'm facing a technical issue. Tone: {sentiment}."
        if entities:
            user_block += f" Related to: {entities}."
        if retrieved_chunks:
            user_block += f"\nHere are some details:\n{knowledge}"
        user_block += f"\n\n{user_message.strip()}"

    elif primary_intent == "complaint":
        user_block = f"{user_prefix} I have a complaint. The tone is {sentiment}."
        if entities:
            user_block += f" It relates to: {entities}."
        user_block += f"\n\n{user_message.strip()}"

    elif primary_intent == "appreciation":
        user_block = f"{user_prefix} I just wanted to say thank you! My message is {sentiment} in tone."
        if entities:
            user_block += f" Specific areas: {entities}."
        user_block += f"\n\n{user_message.strip()}"

    elif primary_intent == "greeting":
        user_block = f"{user_prefix} Hi! Just wanted to say hello. Tone: {sentiment}.\n\n{user_message.strip()}"

    else:
        return None  # Invalid or unsupported intent

    return f"{system_setup}{user_block}\n{assistant_prefix}"


def assess_response_quality(response, retrieved_chunks, user_message):
    """
    Evaluates the quality of a generated response to detect hallucinations and other issues.

    Parameters:
        response (str): The generated response text.
        retrieved_chunks (List[str]): The knowledge chunks used for generation.
        user_message (str): Original user message.

    Returns:
        dict: Assessment results with quality score and flags.
    """
    result = {
        "is_hallucinated": False,
        "is_uncertain": False,
        "is_unsatisfactory": False,
        "quality_score": 0.8,  # Default/starting score
        "feedback": []
    }

    # Convert to lowercase for easier text processing
    response_lower = response.lower()

    # 1. Check for uncertainty markers
    uncertainty_phrases = [
        "i don't know", "i'm not sure", "i am not sure",
        "that's unclear", "i'm uncertain", "i am uncertain",
        "i cannot say", "i can't say", "maybe", "perhaps",
        "it's possible", "it is possible", "i don't have access",
        "i don't have information", "i do not have information"
    ]

    if any(phrase in response_lower for phrase in uncertainty_phrases):
        result["is_uncertain"] = True
        result["quality_score"] -= 0.3
        result["feedback"].append("Response contains uncertainty markers")

    # 2. Check relevance to retrieved knowledge (for factual questions)
    if retrieved_chunks and len(retrieved_chunks) > 0:
        # Create a simple knowledge representation
        knowledge_text = " ".join(retrieved_chunks).lower()
        knowledge_words = set(knowledge_text.split())

        # Check word overlap between response and knowledge
        response_words = set(response_lower.split())
        overlap_count = len(response_words.intersection(knowledge_words))

        # Calculate relevance score
        # Low overlap might indicate hallucination or irrelevance
        relevance_ratio = overlap_count / max(1, min(len(response_words), len(knowledge_words)))

        if relevance_ratio < 0.15:  # Threshold for hallucination detection
            result["is_hallucinated"] = True
            result["quality_score"] -= 0.3
            result["feedback"].append("Low overlap with knowledge base")

    # 3. Check response length (too short responses are often low quality)
    if len(response.split()) < 7:
        result["is_unsatisfactory"] = True
        result["quality_score"] -= 0.2
        result["feedback"].append("Response is too short")

    # 4. Check if response addresses user question (very basic check)
    user_words = set(user_message.lower().split())
    important_user_words = [w for w in user_words if len(w) > 3 and w not in ["what", "when", "where", "which", "how", "would", "could", "should", "about", "with", "this", "that", "there", "their", "have", "your"]]

    # Check if important words from user question appear in response
   # if important_user_words and not any(word in response_lower for word in important_user_words):
    #    result["is_unsatisfactory"] = True
     #   result["quality_score"] -= 0.2
      #  result["feedback"].append("Response doesn't address user's keywords")

    # Final quality determination
    if result["quality_score"] < 0.5:
        result["is_unsatisfactory"] = True

    return result

# Global counter for tracking clarification attempts
clarification_counter = {}

def generate_response(user_id, user_message, sentiment, ner_entities, conversation_history,
                     retrieved_chunks=None, intent_label=None, business_name="your business",
                     clarification_limit=3):
    """
    Comprehensive response generation system with quality assessment and fallbacks.

    Parameters:
        user_id (str): Unique identifier for the user (for tracking clarification attempts).
        user_message (str): The user's input message.
        sentiment (str): Detected sentiment.
        ner_entities (List[str]): Named entities extracted from message.
        conversation_history (List[str]): Previous conversation messages.
        retrieved_chunks (List[str], optional): Knowledge base chunks for answering questions.
        intent_label (str): Detected intent of user's message.
        business_name (str): Company name for personalization.
        clarification_limit (int): Maximum clarification attempts before escalation.

    Returns:
        dict: Response data including text, status, and metadata.
    """
    # Initialize user's clarification counter if not exists
    if user_id not in clarification_counter:
        clarification_counter[user_id] = 0

    # Build the prompt for the model
    prompt = build_godel_prompt(
        user_message=user_message,
        sentiment=sentiment,
        ner_entities=ner_entities,
        conversation_history=conversation_history,
        retrieved_chunks=retrieved_chunks,
        intent_label=intent_label,
        business_name=business_name
    )

    # Get raw model response
    raw_response = call_godel_api(prompt)

    # Assess response quality
    quality_assessment = assess_response_quality(
        response=raw_response,
        retrieved_chunks=retrieved_chunks or [],
        user_message=user_message
    )

    # Handle different quality scenarios
    if quality_assessment["is_unsatisfactory"] or quality_assessment["is_hallucinated"]:
        clarification_counter[user_id] += 1

        # If we've tried to clarify too many times, suggest human support
        if clarification_counter[user_id] >= clarification_limit:
            clarification_counter[user_id] = 0  # Reset counter

            return {
                "text": "I'm sorry, but I'm having trouble understanding your request properly. Would you like me to connect you with a human support agent who can help better?",
                "status": "escalate_to_human",
                "quality_assessment": quality_assessment,
                "clarification_count": clarification_counter[user_id]
            }

        # Otherwise, ask for clarification based on intent
        if intent_label == "support-question":
            clarification_text = "I'm not sure I have the right information to answer that question accurately. Could you please provide more details or rephrase your question?"
        else:
            clarification_text = "I'm sorry, I didn't quite understand what you meant. Could you please elaborate or rephrase that for me?"

        return {
            "text": clarification_text,
            "status": "needs_clarification",
            "quality_assessment": quality_assessment,
            "clarification_count": clarification_counter[user_id]
        }

    # If response is good quality, reset clarification counter and return response
    clarification_counter[user_id] = 0

    # Post-process the response (remove artifacts, fix formatting)
    processed_response = post_process_response(raw_response, business_name)

    return {
        "text": raw_response,
        "status": "success",
        "quality_assessment": quality_assessment,
        "clarification_count": 0
    }

import random
def post_process_response(response, business_name):
    """
    Cleans and improves the raw model response.

    Parameters:
        response (str): Raw model generated text.
        business_name (str): Name of the business for personalization.

    Returns:
        str: Cleaned and improved response.
    """
    # Remove common artifacts that might appear in generated text
    artifacts = [
        "Your response:",
        "AI:",
        "Chatbot:",
        "Assistant:",
        f"{business_name}:",
        "Human:",
        "User:"
    ]

    cleaned = response
    for artifact in artifacts:
        cleaned = cleaned.replace(artifact, "")

    # Fix multiple spaces and line breaks
    cleaned = " ".join(cleaned.split())

    # Ensure the first letter is capitalized
    if cleaned and len(cleaned) > 0:
        cleaned = cleaned[0].upper() + cleaned[1:]

    # Add business name personalization if not present
    if business_name.lower() not in cleaned.lower() and len(cleaned) > 30:
        if "thank you" in cleaned.lower() or "thanks for" in cleaned.lower():
            # Don't add business name if it's already a thank you message
            pass
        elif random.random() < 0.3:  # 30% chance to add business name
            cleaned += f" We at {business_name} are here to help if you need anything else."

    return cleaned.strip()

def prepare_knowledge(chunks):
    """Helper function to prepare knowledge chunks for prompts."""
    if not chunks:
        return "No verified knowledge available - politely ask for more details"
    unique_chunks = list(dict.fromkeys(chunk.strip() for chunk in chunks if chunk.strip()))
    return "\n- ".join([""] + unique_chunks[:3])  # Keep top 3 unique chunks

def chatbot_response(user_message, conversation_context=None, document_data=None, business_name="Your Business"):
    """
    Main chatbot function that integrates all components:
    - Message preprocessing
    - Sentiment analysis
    - Named Entity Recognition
    - Intent recognition
    - Document retrieval
    - Response generation

    Parameters:
        user_message (str): The user's input message
        conversation_context (dict, optional): Previous conversation history and context
        document_data (dict, optional): Preprocessed document data from preprocessing_docs()
        business_name (str): Name of the business for personalization

    Returns:
        dict: {
            'response': str,  # The chatbot's response
            'context': dict,  # Updated conversation context
            'debug_info': dict  # Debug information (intent, sentiment, etc.)
        }
    """
    # Initialize context if not provided
    if conversation_context is None:
        conversation_context = []

    # Initialize document data if not provided
    if document_data is None:
        document_data = {
            'raw_chunks': [],
            'bm25_index': None,
            'tokenized_docs': None,
            'faiss_index': None,
            'document_embeddings': None,
            'intent_labels': []
        }

    # Step 1: Preprocess the user message
    preprocessed_message = preprocess_message(user_message, conversation_context)

    # Step 2: Analyze sentiment
    sentiment_label, sentiment_score = analyze_sentiment(preprocessed_message, conversation_context)
    update_context_with_sentiment(conversation_context, preprocessed_message, sentiment_label)

    # Step 3: Extract named entities
    ner_entities = extract_ner_entities(preprocessed_message, conversation_context)
    entity_words = [entity["word"] for entity in ner_entities]

    # Step 4: Detect intent
    intent_result = detect_intent(
        preprocessed_message["text_use"],
        document_data['intent_labels']
    )

    # Step 5: Check for fallbacks based on intent
    if intent_result['fallback']['status'] == "fallback":
        print("Fallback response debug info:")
        print(f"Intent result: {intent_result}")
        print(f"Sentiment label: {sentiment_label}")
        print(f"NER entities: {ner_entities}")
        print(f"Preprocessed message: {preprocessed_message}")
        
        return {
            'response': intent_result['fallback']['message'],
            'context': conversation_context,
            'debug_info': {
                'intent': intent_result,
                'sentiment': sentiment_label,
                'entities': ner_entities,
                'preprocessed': preprocessed_message
            }
        }

    # Step 6: Retrieve relevant documents if it's a question
    retrieved_chunks = []
    if intent_result['is_question']:
        query_label = intent_result['topic_labels'][0] if intent_result['topic_labels'] else "general"
        retrieved_chunks = hybrid_search(
            preprocessed_message["text_use"],
            query_label,
            document_data['bm25_index'],
            document_data['faiss_index'],
            document_data['raw_chunks'],
            document_data['document_embeddings'],
            document_data['intent_labels'],
            top_n=3
        )

    # Step 7: Format conversation history for the response generator
    history_formatted = []
    for entry in conversation_context[-5:]:  # Get last 5 exchanges
        if "user" in entry:
            history_formatted.append(entry["user"])

    # Step 8: Generate response
    response = generate_response(1,
        user_message=preprocessed_message["original"],
        sentiment=sentiment_label,
        ner_entities=entity_words,
        conversation_history=history_formatted,
        retrieved_chunks=retrieved_chunks,
        intent_label=intent_result['matched_labels'],
        business_name=business_name
    )

    # Step 9: Update context with bot response
    conversation_context.append({"bot": response})

    # Step 10: Return the response and updated context
    return {
        'response': response,
        'context': conversation_context,
        'debug_info': {
            'intent': intent_result,
            'sentiment': sentiment_label,
            'entities': ner_entities,
            'retrieved_chunks': retrieved_chunks,
            'preprocessed': preprocessed_message
        }
    }