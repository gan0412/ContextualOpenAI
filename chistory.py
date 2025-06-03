# chistory.py
from collections import deque
from typing import TypedDict, Literal, Tuple
import json
import os
import tiktoken 

# --- Configuration ---
TOKENIZER_ENCODING_NAME = "cl100k_base"
HISTORY_FILE = "history.json"


class Message(TypedDict):
    """A message in the conversation history."""
    role: Literal["system", "user", "assistant"]
    content: str

def get_token_count_chistory(text: str) -> int:
    """Calculates the number of tokens in a given text string (for history management)."""
    encoding = tiktoken.get_encoding(TOKENIZER_ENCODING_NAME)
    return len(encoding.encode(text))

def load_conversation_state() -> Tuple[deque[Message], int]:
    """
    Loads conversation history and total token count from a JSON file.
    Returns (deque of messages, total token count).
    If file doesn't exist or is empty/corrupt, returns empty deque and 0 tokens.
    """
    if not os.path.exists(HISTORY_FILE):
        # print(f"Starting with empty history.")
        return deque(), 0

    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            messages_list = data.get("messages", [])
            total_tokens = data.get("total_tokens", 0)
            
            history_deque = deque(messages_list)
            # print(f"Loaded {len(history_deque)} messages from '{HISTORY_FILE}'. Total tokens: {total_tokens}")
            return history_deque, total_tokens
    except json.JSONDecodeError:
        print(f"Error decoding JSON from '{HISTORY_FILE}'. Starting with empty history.")
        return deque(), 0
    except Exception as e:
        print(f"An unexpected error occurred loading history from '{HISTORY_FILE}': {e}. Starting with empty history.")
        return deque(), 0

def save_conversation_state(history: deque[Message], total_tokens: int) -> None:
    """
    Saves the current conversation history and total token count to a JSON file.
    """
    data = {
        "messages": list(history),
        "total_tokens": total_tokens
    }
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        # print(f"Saved {len(history)} messages to '{HISTORY_FILE}'. Total tokens: {total_tokens}")
    except Exception as e:
        print(f"An error occurred saving history to '{HISTORY_FILE}': {e}")

def clear_history() -> None:
    """
    Clears the conversation history stored in the history.json file.
    """
    save_conversation_state(deque(), 0)
    print(f"History has been cleared.")
