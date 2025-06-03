# openai/resources/context.py
"""
Context Management Module for Chatbot Conversations

This module provides functionality for managing conversation history and context in a chatbot system,
specifically focusing on token-based context limiting and now with persistence to a file.
It includes features for:
- Loading and saving conversation history from/to a JSON file (now only user messages saved).
- Maintaining conversation history using a deque (unlimited by message count).
- Calculating message relevance using cosine similarity.
- Weighting historical messages based on recency and semantic similarity.
- Formatting conversation context for the LLM with relevance scores.
- Dynamically managing the context window based *solely* on token count to prevent overflow.

The module uses sentence transformers for semantic similarity calculations and implements
a weighted scoring system that combines recency and semantic relevance to determine
which historical messages are most important for the current conversation. It leverages
tiktoken for accurate token counting.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import List
from collections import deque
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken 
import time

# Import functions and types from chistory.py
from chistory import Message, TOKENIZER_ENCODING_NAME, load_conversation_state, save_conversation_state, clear_history # Corrected import for clear_history_file

# Initialize the sentence transformer model with a delay
time.sleep(2)  # Add a 2-second delay before loading the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Constants for LLM token limits (should ideally be configured where LLM is called)
LLM_MAX_CONTEXT_WINDOW = 1000 # Example: Max tokens for a gpt-3.5-turbo context window
LLM_MIN_CONTEXT_WINDOW = 200  # Minimum useful context size

# --- NEW CONSTANT FOR EXPONENTIAL DECAY ---
RECENCY_DECAY_ALPHA = 0.80 # Decay factor: 0 < alpha < 1. Closer to 1 is slower decay.
                           # Experiment with this value (e.g., 0.9, 0.7, 0.5)

# --- NEW CONSTANT FOR EXPONENTIAL SEMANTIC SIMILARITY ---
SEMANTIC_DECAY_BETA = 3.0  # Multiplier for raw similarity before exponentiation. Higher value means more aggressive focus on high similarity.
# ------------------------------------------

def get_token_count(text: str) -> int:
    """Calculates the number of tokens in a given text string using tiktoken."""
    encoding = tiktoken.get_encoding(TOKENIZER_ENCODING_NAME)
    return len(encoding.encode(text))


def get_cosine_similarity(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two text strings.
    
    Args:
        text1: First text string to compare
        text2: Second text string to compare
        
    Returns:
        float: Cosine similarity score between -1 and 1 (often 0-1 for common embeddings),
               where 1 means identical.
    """
    embedding1 = model.encode(text1)
    embedding2 = model.encode(text2)
    
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return float(similarity)


def contextualize_prompt(
    prompt: str,
    max_tokens_limit: int # This is the target context size for the LLM
) -> List[Message]:
    """
    Processes a user prompt with conversation history context, respecting token limits.
    
    This function:
    1. Loads conversation history (all messages) from a file.
    2. Adds the current user's message to a *working copy* of the history.
    3. Manages token limits on this working copy (evicting oldest if needed).
    4. Calculates relevance scores for historical messages in the working copy.
    5. Combines recency and semantic similarity weights.
    6. Returns a list of messages with weighted historical context for the LLM.
    7. **Saves only the user messages** from the working copy back to the file.
    
    Args:
        prompt: The current user message to process.
        max_tokens_limit: The maximum number of tokens allowed for the historical messages
                          and the current user message. The system message tokens will be
                          added on top of this for the total LLM context.
        
    Returns:
        List[Message]: Messages to be sent to the LLM (includes system instruction,
                             weighted historical messages, and current user message).
    """
    # Load current *full* history from file (it will contain user messages only if we filter it on save)
    # This deque will be used to build the LLM's context.
    conversation_history_for_llm_context, current_total_tokens_for_llm_context = load_conversation_state()

    # Define the system instruction message and its token count
    system_instruction_content = """You are a helpful AI assistant.
Below is the previous conversation history, ordered from oldest to newest.
Each historical message is prefixed with a numerical score in brackets, e.g., [0.85].
This score indicates the message's relevance and recency, with higher scores meaning more importance.
Please pay more attention to messages with higher scores when formulating your response to the current user query."""
    system_message: Message = {"role": "system", "content": system_instruction_content}
    system_message_tokens = get_token_count(system_message["content"])


    # The effective maximum context window for the LLM, including the system message.
    # This is what LLM_MAX_CONTEXT_WINDOW refers to.
    effective_llm_context_window = max_tokens_limit + system_message_tokens 

    # Validate max_tokens_limit against the LLM's actual maximum context window
    # The sum of max_tokens_limit (for history+user) and system_message_tokens must not exceed LLM_MAX_CONTEXT_WINDOW
    if effective_llm_context_window > LLM_MAX_CONTEXT_WINDOW:
        print(f"Warning: The sum of provided max_tokens_limit ({max_tokens_limit}) and system message tokens ({system_message_tokens}) exceeds LLM_MAX_CONTEXT_WINDOW ({LLM_MAX_CONTEXT_WINDOW}). Adjusting max_tokens_limit down to {LLM_MAX_CONTEXT_WINDOW - system_message_tokens}.")
        max_tokens_limit = LLM_MAX_CONTEXT_WINDOW - system_message_tokens
        # Ensure it doesn't go below MIN_CONTEXT_WINDOW after adjustment
        if max_tokens_limit < LLM_MIN_CONTEXT_WINDOW:
            max_tokens_limit = LLM_MIN_CONTEXT_WINDOW
            print(f"Further adjustment: max_tokens_limit set to minimum useful context ({LLM_MIN_CONTEXT_WINDOW}).")
    elif effective_llm_context_window < LLM_MIN_CONTEXT_WINDOW:
        print(f"Warning: The combined context ({effective_llm_context_window}) is too small. Adjusting max_tokens_limit up to {LLM_MIN_CONTEXT_WINDOW - system_message_tokens}.")
        max_tokens_limit = LLM_MIN_CONTEXT_WINDOW - system_message_tokens
        if max_tokens_limit < 0: # Ensure max_tokens_limit for history isn't negative
            max_tokens_limit = 0
            print("Warning: max_tokens_limit for history set to 0 as system message already takes up too much space for minimum context.")


    current_message: Message = {"role": "user", "content": prompt}
    current_message_tokens = get_token_count(current_message["content"])

    # --- Add current user message to the working history deque ---
    # This deque will be used to determine what goes to the LLM.
    conversation_history_for_llm_context.append(current_message)
    current_total_tokens_for_llm_context += current_message_tokens


    # --- Token Management: Evict oldest messages if adding current message exceeds limit ---
    # Projected tokens now correctly includes system message tokens for the *overall* context check
    projected_tokens = current_total_tokens_for_llm_context + system_message_tokens 

    # print("projected_tokens:", projected_tokens)

    # Remove oldest messages from history until the *total context* (history + system + current user) fits within LLM_MAX_CONTEXT_WINDOW
    while projected_tokens > max_tokens_limit and len(conversation_history_for_llm_context) > 0:
        # If the oldest message is the one we just added (current user), and it still exceeds limit, handle it
        if len(conversation_history_for_llm_context) == 1 and conversation_history_for_llm_context[0]["role"] == "user":
            # This means the current user message itself is too long even with an empty history and system message.
            print(f"Warning: Current user prompt ({current_message_tokens} tokens) alone plus system message ({system_message_tokens} tokens) exceeds LLM_MAX_CONTEXT_WINDOW ({LLM_MAX_CONTEXT_WINDOW}). History will be empty. Returning only system and current prompt.")
            save_conversation_state(deque(), 0) # Save empty history to file
            return [system_message, current_message] 
        
        removed_message = conversation_history_for_llm_context.popleft() # Remove from the left (oldest)
        current_total_tokens_for_llm_context -= get_token_count(removed_message["content"])
        projected_tokens = current_total_tokens_for_llm_context + system_message_tokens 
    
    # --- REMOVED THE PROBLEMATIC `if` BLOCK HERE ---
    # The `while` loop above now correctly handles trimming the history.
    # The previous `if` statement here was too aggressive and could clear history prematurely.
    # ------------------------------------------------


    # --- Construct the contextualized prompt for the LLM ---
    weighted_messages: List[Message] = [system_message] # Start with the system instruction
    
    # Get a copy of the current history *for LLM context* (don't modify deque during iteration)
    history_list_for_llm = list(conversation_history_for_llm_context) 

    # The current user message is always the last element in history_list_for_llm now
    # We need to exclude it from historical weighting and add it at the end.
    messages_for_weighting = history_list_for_llm[:-1] # All messages except the last one (current user)
    
    if not messages_for_weighting:
        pass 
    else:
        # --- Calculate and combine weights ---
        recency_weights: List[float] = [
            RECENCY_DECAY_ALPHA**(len(messages_for_weighting) - i - 1)
            for i in range(len(messages_for_weighting))
        ]
        
        similarity_weights: List[float] = []
        for message in messages_for_weighting:
            raw_similarity = get_cosine_similarity(current_message["content"], message["content"])
            # --- MODIFIED: Apply exponential relationship to semantic similarity ---
            # Ensures non-negative similarity and applies the exponential boost
            transformed_similarity = pow(max(0, raw_similarity) * SEMANTIC_DECAY_BETA, 2) 
            similarity_weights.append(transformed_similarity)
            # --------------------------------------------------------------------
        
        if similarity_weights:
            total_similarity = sum(similarity_weights)
            if total_similarity > 0:
                similarity_weights = [w / total_similarity for w in similarity_weights]
            else:
                # Fallback if all transformed similarities are zero (e.g., all raw similarities were negative or zero)
                similarity_weights = [1.0 / len(similarity_weights)] * len(similarity_weights)
        
        combined_weights: List[float] = []
        total_recency_weight_sum = sum(recency_weights)
        for i in range(len(messages_for_weighting)):
            recency_weight_normalized = recency_weights[i] / total_recency_weight_sum if total_recency_weight_sum > 0 else 0
            
            # Semantic similarity will now have a 70% influence, recency 30%
            combined_weight = 0.3 * recency_weight_normalized + 0.7 * similarity_weights[i]
            # You can experiment with these values (e.g., 0.2 and 0.8, or 0.1 and 0.9)
            # to fine-tune the balance between recency and semantic relevance.
            # Just ensure their sum is 1.0.
            
            combined_weights.append(combined_weight)
        
        # --- Use numerical tags for historical messages ---
        weighted_history_items = []
        for i, message in enumerate(messages_for_weighting):
            weighted_history_items.append((combined_weights[i], message))

        weighted_history_items.sort(key=lambda x: x[0], reverse=True)

        for weight, message in weighted_history_items:
            weight_tag = f"[{weight:.2f}]" 
            weighted_messages.append({"role": message["role"], "content": f"{weight_tag} {message['content']}"})
        
    # Add current user message at the end of the context for the LLM
    weighted_messages.append(current_message)
    
    # --- Filter for saving: ONLY USER MESSAGES ---
    user_messages_for_saving = deque()
    total_tokens_for_saving = 0
    for msg in conversation_history_for_llm_context:
        if msg["role"] == "user":
            user_messages_for_saving.append(msg)
            total_tokens_for_saving += get_token_count(msg["content"])
    
    # Save only the user messages history to the file
    save_conversation_state(user_messages_for_saving, total_tokens_for_saving)
    
    # The print statements are commented out as per previous instructions.

    time.sleep(2) # This sleep is not related to the core logic, keeping it as is.
    
    return weighted_messages