import logging
import time
import traceback

import copy
from api.utilities.intent_recognition import detect_intent
from api.utilities.message_processing import preprocess_message
from api.utilities.ner import extract_ner_entities
from api.utilities.retrieval_system import hybrid_search
from api.utilities.sentiment_analysis import analyze_sentiment, update_context_with_sentiment
from api.services.nlp_manager import NLPManager
from transformers import StoppingCriteria, StoppingCriteriaList
import replicate

# Configure logger
logger = logging.getLogger(__name__)

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        for stop_seq in self.stop_token_ids:
            if input_ids[0][-len(stop_seq):].tolist() == stop_seq:
                return True
        return False

def format_conversation_history(history_list):
    """
    Formats a conversation history list of dicts into LLaMA-3 chat-style prompt blocks.
    Expects history like: [{'user': '...'}, {'bot': '...'}, ...]
    """
    formatted = []
    for turn in history_list:
        if 'user' in turn:
            formatted.append(
                "<|start_header_id|>user<|end_header_id|>\n\n"
                f"{turn['user'].strip()}\n<|eot_id|>"
            )
        elif 'bot' in turn:
            formatted.append(
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{turn['bot'].strip()}\n<|eot_id|>"
            )
    return "\n".join(formatted)

def call_godel_api(prompt, system_prompt):
    logger.warning("call_godel_api started")
    logger.debug(f"Prompt preview: {prompt[:200]}...")
    start_time = time.time()

    try:
        # Since the prompt is fully formed, we use the simplest passthrough template
        input_data = {
            "prompt": prompt,
            "system_prompt": system_prompt,  # Still passed as required by some APIs
            "max_new_tokens": 512,
            "temperature": 0.4,
            "top_p": 0.9,
            "do_sample": True,
            "prompt_template": "{prompt}"
        }

        print("---------------START SYSTEM PROMPT-----------------")
        print(system_prompt)
        print("---------------END SYSTEM PROMPT-----------------")
        print("---------------START PROMPT-----------------")
        print(prompt)
        print("---------------END PROMPT-----------------")

        logger.warning("Sending request to Replicate API")
        output = replicate.run("meta/meta-llama-3-8b-instruct", input=input_data)

        response = "".join(output)
        logger.warning("Received response from Replicate")
        logger.debug(f"Model response: {response[:300]}...")

        end_time = time.time()
        logger.warning(f"call_godel_api completed in {end_time - start_time:.3f}s")
        return response

    except Exception as e:
        end_time = time.time()
        traceback.print_exc()
        logger.error(f"call_godel_api failed after {end_time - start_time:.3f}s: {str(e)}")
        logger.error(traceback.format_exc())
        return "Error generating response."
    
def build_godel_prompt(user_message, sentiment, ner_entities, conversation_history, retrieved_chunks=None, intent_label=None, business_name="your business"):
    """
    Builds a prompt for decoder-only models (e.g., TinyLLaMA) using sentiment, NER, intent, retrieved knowledge,
    and a list of previous user messages (user-only history).
    """
    logger.warning("build_godel_prompt started")
    logger.warning(f"User message: '{user_message[:100]}...'")
    logger.warning(f"Sentiment: {sentiment}")
    logger.warning(f"NER entities: {ner_entities}")
    logger.warning(f"Intent label: {intent_label}")
    logger.warning(f"Business name: {business_name}")
    logger.warning(f"Retrieved chunks count: {len(retrieved_chunks) if retrieved_chunks else 0}")
    logger.warning(f"Conversation history count: {len(conversation_history) if conversation_history else 0}")
    
    start_time = time.time()

    valid_intents = ["support-question", "support", "complaint", "appreciation", "greeting"]
    if isinstance(intent_label, list):
        primary_intent = next((i.lower() for i in intent_label if i.lower() in valid_intents), None)
        logger.warning(f"Processing intent list: {intent_label} -> primary: {primary_intent}")
    else:
        primary_intent = intent_label.lower() if intent_label and intent_label.lower() in valid_intents else None
        logger.warning(f"Using single intent: {intent_label} -> primary: {primary_intent}")

    # Prepare knowledge and entity info
    logger.warning("Preparing knowledge and entity information")
    entities = ", ".join(ner_entities) if ner_entities else None
    logger.debug(f"Prepared knowledge: {retrieved_chunks}")
    logger.debug(f"Entity string: {entities}")

    system_setup = f"""
    You are a confident and professional customer service agent at {business_name}.
    Speak as the company — never mention documents, internal files, or sources.

    Always respond in first-person, as part of the business.

    Keep answers short, clear, and direct. Do not over-explain or repeat.
    Avoid lists unless necessary. If used, keep them brief.

    Never say things like "Based on the document..." or "It mentions...".
    You are the company — answer with full authority.
    """
    formatted_history = format_conversation_history(conversation_history[:-1])

    # Inject into system prompt to guide the assistant's thinking
    system_setup += f"""

    The following is the prior chat history between the user and assistant.
    Use it to help answer the next message, but never refer to it explicitly.

    {formatted_history}
    """
    print("---------------START HISTORY-----------------")
    print(conversation_history)
    print("---------------END HISTORY-----------------")

    # Compose current user turn based on intent
    logger.warning(f"Composing user block for intent: {primary_intent}")
    if primary_intent == "support-question":
        user_block = f"I need help with the following question. My sentiment is {sentiment}."
        if entities:
            user_block += f" This involves: {entities}."
        if retrieved_chunks:
            user_block += f"\nHere is some background info:\n{retrieved_chunks} (YOUR ANSWER HAVE TO BE BASED ON THIS INFO ONLY)"
        user_block += f"\n\n{user_message.strip()}"
        logger.warning("Built support-question user block")

    elif primary_intent == "support":
        user_block = f"I'm facing a technical issue. Tone: {sentiment}."
        if entities:
            user_block += f" Related to: {entities}."
        if retrieved_chunks:
            user_block += f"\nHere are some details:\n{retrieved_chunks} (YOUR ANSWER HAVE TO BE BASED ON THIS INFO ONLY)"
        user_block += f"\n\n{user_message.strip()}"
        logger.warning("Built support user block")

    elif primary_intent == "complaint":
        user_block = f"I have a complaint. The tone is {sentiment}."
        if entities:
            user_block += f" It relates to: {entities}."
        user_block += f"\n\n{user_message.strip()}"
        logger.warning("Built complaint user block")

    elif primary_intent == "appreciation":
        user_block = f"I just wanted to say thank you! My message is {sentiment} in tone."
        if entities:
            user_block += f" Specific areas: {entities}."
        user_block += f"\n\n{user_message.strip()}"
        logger.warning("Built appreciation user block")

    elif primary_intent == "greeting":
        user_block = f"Hi! Just wanted to say hello. Tone: {sentiment}.\n\n{user_message.strip()}"
        logger.warning("Built greeting user block")

    else:
        logger.warning(f"Unknown intent: {primary_intent}, returning None")
        return None  # Unknown intent

    # Final prompt
    logger.warning("Building final prompt")
    formatted_history = format_conversation_history(conversation_history[:-1])

    final_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_setup.strip()}"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_block.strip()}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    end_time = time.time()
    logger.warning(f"build_godel_prompt completed in {end_time - start_time:.3f}s")
    logger.debug(f"Final prompt preview: {final_prompt[:300]}...")
    return final_prompt, system_setup


def assess_response_quality(response, retrieved_chunks, user_message):
    """
    Evaluates the quality of a generated response to detect hallucinations and other issues.
    """
    logger.warning("assess_response_quality started")
    logger.debug(f"Assessing response: '{response[:100]}...'")
    logger.debug(f"User message: '{user_message[:100]}...'")
    logger.warning(f"Retrieved chunks count: {len(retrieved_chunks) if retrieved_chunks else 0}")
    
    start_time = time.time()
    
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
    logger.warning("Checking for uncertainty markers")
    uncertainty_phrases = [
        "i don't know", "i'm not sure", "i am not sure",
        "that's unclear", "i'm uncertain", "i am uncertain",
        "i cannot say", "i can't say", "maybe", "perhaps",
        "it's possible", "it is possible", "i don't have access",
        "i don't have information", "i do not have information"
    ]

    uncertainty_found = [phrase for phrase in uncertainty_phrases if phrase in response_lower]
    if uncertainty_found:
        result["is_uncertain"] = True
        result["quality_score"] -= 0.3
        result["feedback"].append("Response contains uncertainty markers")
        logger.warning(f"Uncertainty markers found: {uncertainty_found}")
    else:
        logger.warning("No uncertainty markers found")

    # 2. Check relevance to retrieved knowledge (for factual questions)
    if retrieved_chunks and len(retrieved_chunks) > 0:
        logger.warning("Checking relevance to knowledge base")
        # Create a simple knowledge representation
        knowledge_text = " ".join(retrieved_chunks).lower()
        knowledge_words = set(knowledge_text.split())

        # Check word overlap between response and knowledge
        response_words = set(response_lower.split())
        overlap_count = len(response_words.intersection(knowledge_words))

        # Calculate relevance score
        # Low overlap might indicate hallucination or irrelevance
        relevance_ratio = overlap_count / max(1, min(len(response_words), len(knowledge_words)))
        logger.warning(f"Knowledge relevance ratio: {relevance_ratio:.3f} (overlap: {overlap_count})")

        if relevance_ratio < 0.15:  # Threshold for hallucination detection
            result["is_hallucinated"] = True
            result["quality_score"] -= 0.3
            result["feedback"].append("Low overlap with knowledge base")
            logger.warning(f"Low knowledge overlap detected: {relevance_ratio:.3f}")
        else:
            logger.warning("Good knowledge relevance")
    else:
        logger.warning("No retrieved chunks to check relevance against")

    # 3. Check response length (too short responses are often low quality)
    response_word_count = len(response.split())
    logger.warning(f"Response word count: {response_word_count}")
    if response_word_count < 1:
        result["is_unsatisfactory"] = True
        result["quality_score"] -= 0.2
        result["feedback"].append("Response is too short")
        logger.warning("Response is too short")

    # 4. Check if response addresses user question (very basic check)
    user_words = set(user_message.lower().split())
    important_user_words = [w for w in user_words if len(w) > 3 and w not in ["what", "when", "where", "which", "how", "would", "could", "should", "about", "with", "this", "that", "there", "their", "have", "your"]]
    logger.debug(f"Important user words: {important_user_words}")

    # Final quality determination
    if result["quality_score"] < 0.3:
        result["is_unsatisfactory"] = True
        logger.warning("Overall quality score too low, marking as unsatisfactory")

    end_time = time.time()
    logger.warning(f"assess_response_quality completed in {end_time - start_time:.3f}s")
    logger.warning(f"Quality assessment result: {result}")
    return result

# Global counter for tracking clarification attempts
clarification_counter = {}

def generate_response(user_id, user_message, sentiment, ner_entities, conversation_history,
                     retrieved_chunks=None, intent_label=None, business_name="your business",
                     clarification_limit=3):
    """
    Comprehensive response generation system with quality assessment and fallbacks.
    """
    logger.warning(f"generate_response started for user: {user_id}")
    logger.warning(f"User message: '{user_message[:100]}...'")
    logger.warning(f"Sentiment: {sentiment}")
    logger.warning(f"NER entities: {ner_entities}")
    logger.warning(f"Intent label: {intent_label}")
    logger.warning(f"Business name: {business_name}")
    logger.warning(f"Retrieved chunks count: {len(retrieved_chunks) if retrieved_chunks else 0}")
    logger.warning(f"Conversation history count: {len(conversation_history) if conversation_history else 0}")
    logger.warning(f"Clarification limit: {clarification_limit}")
    
    start_time = time.time()
    
    # Initialize user's clarification counter if not exists
    if user_id not in clarification_counter:
        clarification_counter[user_id] = 0
        logger.warning(f"Initialized clarification counter for user {user_id}")
    
    current_count = clarification_counter[user_id]
    logger.warning(f"Current clarification count for user {user_id}: {current_count}")

    # Build the prompt for the model
    logger.warning("Building prompt for model")
    prompt_start = time.time()
    prompt, system_prompt = build_godel_prompt(
        user_message=user_message,
        sentiment=sentiment,
        ner_entities=ner_entities,
        conversation_history=conversation_history,
        retrieved_chunks=retrieved_chunks,
        intent_label=intent_label,
        business_name=business_name
    )
    prompt_end = time.time()
    logger.warning(f"Prompt building completed in {prompt_end - prompt_start:.3f}s")

    # Get raw model response
    logger.warning("Getting raw model response")
    api_start = time.time()
    raw_response = call_godel_api(prompt, system_prompt)
    api_end = time.time()
    logger.warning(f"Model API call completed in {api_end - api_start:.3f}s")

    # Assess response quality
    logger.warning("Assessing response quality")
    quality_start = time.time()
    quality_assessment = assess_response_quality(
        response=raw_response,
        retrieved_chunks=retrieved_chunks or [],
        user_message=user_message
    )
    quality_end = time.time()
    logger.warning(f"Quality assessment completed in {quality_end - quality_start:.3f}s")

    # Handle different quality scenarios
    if quality_assessment["is_unsatisfactory"] or quality_assessment["is_hallucinated"]:
        logger.warning("Response quality issues detected, incrementing clarification counter")
        clarification_counter[user_id] += 1
        new_count = clarification_counter[user_id]
        logger.warning(f"Clarification count for user {user_id} increased to: {new_count}")

        # If we've tried to clarify too many times, suggest human support
        if clarification_counter[user_id] >= clarification_limit:
            logger.warning(f"Clarification limit reached for user {user_id}, escalating to human")
            clarification_counter[user_id] = 0  # Reset counter
            logger.warning(f"Reset clarification counter for user {user_id}")

            end_time = time.time()
            logger.warning(f"generate_response completed (escalation) in {end_time - start_time:.3f}s")
            return {
                "text": "I'm sorry, but I'm having trouble understanding your request properly. Would you like me to connect you with a human support agent who can help better?",
                "status": "escalate_to_human",
                "quality_assessment": quality_assessment,
                "clarification_count": 0
            }

        # Otherwise, ask for clarification based on intent
        logger.warning(f"Asking for clarification based on intent: {intent_label}")
        if intent_label == "support-question":
            clarification_text = "I'm not sure I have the right information to answer that question accurately. Could you please provide more details or rephrase your question?"
        else:
            clarification_text = "I'm sorry, I didn't quite understand what you meant. Could you please elaborate or rephrase that for me?"

        end_time = time.time()
        logger.warning(f"generate_response completed (clarification) in {end_time - start_time:.3f}s")
        return {
            "text": clarification_text,
            "status": "needs_clarification",
            "quality_assessment": quality_assessment,
            "clarification_count": clarification_counter[user_id]
        }

    # If response is good quality, reset clarification counter and return response
    logger.warning("Response quality acceptable, resetting clarification counter")
    clarification_counter[user_id] = 0

    # Post-process the response (remove artifacts, fix formatting)
    logger.warning("Post-processing response")
    process_start = time.time()
    processed_response = post_process_response(raw_response, business_name)
    process_end = time.time()
    logger.warning(f"Post-processing completed in {process_end - process_start:.3f}s")
    logger.warning(f"Final processed response: '{processed_response[:100]}...'")

    end_time = time.time()
    logger.warning(f"generate_response completed successfully in {end_time - start_time:.3f}s")
    return {
        "text": processed_response,
        "status": "success",
        "quality_assessment": quality_assessment,
        "clarification_count": 0
    }

import random
def post_process_response(response, business_name):
    """
    Extracts only the first Assistant response and cleans it.
    """
    logger.warning("post_process_response started")
    logger.debug(f"Input response: '{response[:200]}...'")
    logger.debug(f"Business name: {business_name}")
    
    start_time = time.time()
    original_response = response

    # Step 1: Extract only the first reply after "Assistant:"
    logger.warning("Extracting Assistant response")
    if "Assistant:" in response:
        response = response.split("Assistant:", 1)[1]
        logger.debug("Found and extracted Assistant section")
        # Stop at the next "User:" or "Assistant:" if exists
        for stop_token in ["User:", "Assistant:"]:
            if stop_token in response:
                response = response.split(stop_token, 1)[0]
                logger.debug(f"Stopped at {stop_token}")
                break

    # Step 2: Remove common artifacts
    logger.warning("Removing common artifacts")
    artifacts = [
        "Your response:",
        "AI:",
        "Chatbot:",
        f"{business_name}:",
        "Human:",
        "User:",
        "Assistant:"  # Just in case it remains
    ]

    cleaned = response
    removed_artifacts = []
    for artifact in artifacts:
        if artifact in cleaned:
            cleaned = cleaned.replace(artifact, "")
            removed_artifacts.append(artifact)
    
    if removed_artifacts:
        logger.debug(f"Removed artifacts: {removed_artifacts}")

    # Step 3: Normalize whitespace
    logger.warning("Normalizing whitespace")
    cleaned = " ".join(cleaned.split())

    # Step 4: Capitalize
    if cleaned and len(cleaned) > 0:
        cleaned = cleaned[0].upper() + cleaned[1:]
        logger.debug("Capitalized first letter")

    # Step 5: Optional personalization
    logger.warning("Applying optional personalization")
    if business_name.lower() not in cleaned.lower() and len(cleaned) > 30:
        if "thank you" not in cleaned.lower() and "thanks for" not in cleaned.lower():
            if random.random() < 0.3:
                cleaned += f" We at {business_name} are here to help if you need anything else."
                logger.debug("Added business personalization")

    end_time = time.time()
    logger.warning(f"post_process_response completed in {end_time - start_time:.3f}s")
    logger.debug(f"Final cleaned response: '{cleaned}'")
    return cleaned.strip()

def prepare_knowledge(chunks):
    """Helper function to prepare knowledge chunks for prompts."""
    if not chunks:
        return "No verified knowledge available - politely ask for more details"
    unique_chunks = list(dict.fromkeys(chunk.strip() for chunk in chunks if chunk.strip()))
    MAX_CHUNK_LEN = 2000  # characters, or count tokens using a tokenizer

    clipped_chunks = [chunk[:MAX_CHUNK_LEN] for chunk in unique_chunks[:3]]
    return "\n- " + "\n- ".join(clipped_chunks)

def chatbot_response(user_message, conversation_context=None, document_data=None, business_name="Your Business"):
    """
    Main chatbot function that integrates all components:
    - Message preprocessing
    - Sentiment analysis
    - Named Entity Recognition
    - Intent recognition
    - Document retrieval
    - Response generation
    """
    logger.warning("=== CHATBOT_RESPONSE STARTED ===")
    logger.warning(f"User message: '{user_message[:100]}...'")
    logger.warning(f"Business name: {business_name}")
    logger.warning(f"Conversation context entries: {len(conversation_context) if conversation_context else 0}")
    logger.warning(f"Document data provided: {'Yes' if document_data else 'No'}")
    
    start_time = time.time()

    # Initialize context if not provided
    if conversation_context is None:
        conversation_context = []
        logger.warning("Initialized empty conversation context")
    previous_messages = copy.deepcopy(conversation_context)
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
        logger.warning("Initialized empty document data")
    else:
        logger.warning("Document data summary:")
        logger.warning(f"- Raw chunks: {len(document_data.get('raw_chunks', []))}")
        logger.warning(f"- BM25 index: {'Available' if document_data.get('bm25_index') else 'None'}")
        logger.warning(f"- FAISS index: {'Available' if document_data.get('faiss_index') else 'None'}")
        logger.warning(f"- Document embeddings: {'Available' if document_data.get('document_embeddings') is not None else 'None'}")
        logger.warning(f"- Intent labels: {len(document_data.get('intent_labels', []))}")

    # Step 1: Preprocess the user message
    logger.warning("STEP 1: Preprocessing user message")
    preprocess_start = time.time()
    preprocessed_message = preprocess_message(user_message, conversation_context)
    preprocess_end = time.time()
    logger.warning(f"Message preprocessing completed in {preprocess_end - preprocess_start:.3f}s")
    logger.debug(f"Preprocessed result: {preprocessed_message}")

    # Step 2: Analyze sentiment
    logger.warning("STEP 2: Analyzing sentiment")
    sentiment_start = time.time()
    sentiment_label, sentiment_score = analyze_sentiment(preprocessed_message, conversation_context)
    update_context_with_sentiment(conversation_context, preprocessed_message, sentiment_label)
    sentiment_end = time.time()
    logger.warning(f"Sentiment analysis completed in {sentiment_end - sentiment_start:.3f}s")
    logger.warning(f"Sentiment: {sentiment_label} (score: {sentiment_score:.3f})")
    # Step 3: Extract named entities
    logger.warning("STEP 3: Extracting named entities")
    ner_start = time.time()
    ner_entities = extract_ner_entities(preprocessed_message, conversation_context)
    entity_words = [entity["word"] for entity in ner_entities]
    ner_end = time.time()
    logger.warning(f"NER extraction completed in {ner_end - ner_start:.3f}s")
    logger.warning(f"Extracted {len(ner_entities)} entities: {entity_words}")


    # Step 4: Detect intent
    logger.warning("STEP 4: Detecting intent")
    intent_start = time.time()
    intent_result = detect_intent(
        preprocessed_message["text_use"],
        document_data['intent_labels']
    )
    intent_end = time.time()
    logger.warning(f"Intent detection completed in {intent_end - intent_start:.3f}s")
    logger.warning(f"Intent result: {intent_result}")

    # Step 5: Check for fallbacks based on intent
    logger.warning("STEP 5: Checking for fallbacks")
    if intent_result['fallback']['status'] == "fallback":
        logger.warning("Fallback response triggered")
        logger.warning("Fallback response debug info:")
        logger.warning(f"Intent result: {intent_result}")
        logger.warning(f"Sentiment label: {sentiment_label}")
        logger.warning(f"NER entities: {ner_entities}")
        logger.warning(f"Preprocessed message: {preprocessed_message}")
        
        end_time = time.time()
        logger.warning(f"=== CHATBOT_RESPONSE COMPLETED (FALLBACK) in {end_time - start_time:.3f}s ===")
        
        return {
            'response': {'text': intent_result['fallback']['message']},
            'context': conversation_context,
            'debug_info': {
                'intent': intent_result,
                'sentiment': sentiment_label,
                'entities': ner_entities,
                'preprocessed': preprocessed_message
            }
        }

    # Step 6: Retrieve relevant documents if it's a question
    logger.warning("STEP 6: Document retrieval")
    retrieved_chunks = []
    if intent_result['is_question']:
        logger.warning("Question detected, performing document retrieval")
        retrieval_start = time.time()
        query_label = intent_result['topic_labels'][0] if intent_result['topic_labels'] else "general"
        logger.warning(f"Using query label: {query_label}")
        
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
        retrieval_end = time.time()
        logger.warning(f"Document retrieval completed in {retrieval_end - retrieval_start:.3f}s")
        logger.warning(f"Retrieved {len(retrieved_chunks)} chunks")
        logger.debug(f"Retrieved chunks preview: {[chunk[:100] + '...' for chunk in retrieved_chunks[:2]]}")
    else:
        logger.warning("Not a question, skipping document retrieval")

    # Step 8: Generate response
    logger.warning("STEP 8: Generating response")
    response_start = time.time()
    response = generate_response(
        user_id=1,  # Default user ID
        user_message=preprocessed_message["original"],
        sentiment=sentiment_label,
        ner_entities=entity_words,
        conversation_history=previous_messages,
        retrieved_chunks=retrieved_chunks,
        intent_label=intent_result['matched_labels'],
        business_name=business_name
    )
    response_end = time.time()
    logger.warning(f"Response generation completed in {response_end - response_start:.3f}s")
    logger.warning(f"Response status: {response.get('status')}")
    # Step 9: Update context with bot response
    logger.warning("STEP 9: Updating context with bot response")
    conversation_context.append({"bot": response['text']})

    # Step 10: Return the response and updated context
    end_time = time.time()
    total_time = end_time - start_time
    logger.warning(f"=== CHATBOT_RESPONSE COMPLETED SUCCESSFULLY in {total_time:.3f}s ===")
    
    return {
        'response': response,
        'context': conversation_context,
        'debug_info': {
            'intent': intent_result,
            'sentiment': sentiment_label,
            'entities': ner_entities,
            'retrieved_chunks': retrieved_chunks,
            'preprocessed': preprocessed_message,
            'processing_times': {
                'total': total_time,
                'preprocessing': preprocess_end - preprocess_start,
                'sentiment': sentiment_end - sentiment_start,
                'ner': ner_end - ner_start,
                'intent': intent_end - intent_start,
                'retrieval': retrieval_end - retrieval_start if intent_result['is_question'] else 0,
                'response_generation': response_end - response_start
            }
        }
    }