# ContextualOpenAI

If you want to make your OpenAI generate context-based responses and copy the necessary files in this repository, follow these simple steps
1. Navigate to the directory where you want to copy the files and enable context-based responses 
2. Copy and paste the following script:
(For linux/unix users)
  ```bash
    git clone --depth 1 --filter=blob:none --sparse https://github.com/gan0412/ContextualOpenAI.git temp_ContextualOpenAI
    cd temp_ContextualOpenAI
    git sparse-checkout set context.py chistory.py
    mv context.py chistory.py ..
    cd ..
    rm -rf temp_ContextualOpenAI
    echo "Files have been copied to the current directory."
  ```
(For powershell users)
```bash
  git clone --depth 1 --filter=blob:none --sparse https://github.com/gan0412/ContextualOpenAI.git temp_ContextualOpenAI
  Set-Location temp_ContextualOpenAI
  git sparse-checkout set context.py chistory.py
  Move-Item context.py -Destination ..
  Move-Item chistory.py -Destination ..
  Set-Location ..
  Remove-Item temp_ContextualOpenAI -Recurse -Force
  Write-Host "Files have been copied to the current directory."
```



---


## How to Use `context.py` Functions for Context-Based OpenAI Responses

The core functionality for enabling context-based responses is provided by two functions:

- `contextualize_prompt(prompt: str, max_tokens_limit: int) -> List[Message]`
- `clear_history()`


### 1. Generating Contextual Responses

Use `contextualize_prompt` to prepare your prompt with relevant conversation history for the OpenAI API. The parameters are:    
- user_input: the client's current prompt
- max_tokens_limit: maximum token length of the context window 

**Use Case:**
`contextualize_prompt` should be called every time you want to generate a context-aware response.

**Example Usage:**

```python
from context import contextualize_prompt

# Your current user prompt
user_input = "Tell me more about the last topic we discussed."

# Set a max token limit for the context window (e.g., 800)
context_messages = contextualize_prompt(user_input, max_tokens_limit=800)

# Send this message list to OpenAI's chat/completions endpoint
# For example, with openai-python:
import openai

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=context_messages
)

print(response['choices'][0]['message']['content'])
```

- The function automatically loads previous user messages from `history.json`, manages the token window, and attaches relevance scores to historical messages.


### 2. Clearing Conversation History

Use `clear_history` to reset or clear the stored conversation history (e.g., at the start of a new session).

**Example Usage:**

```python
from chistory import clear_history

# This will clear history.json and start fresh
clear_history()
```

---

## Notes

- 
- Only user messages are persisted; system and assistant messages are handled internally for context.
- The token limit ensures that the total context fits within your OpenAI model's window.
- Make sure to have the required dependencies installed (`sentence-transformers`, `tiktoken`, `scikit-learn`).
