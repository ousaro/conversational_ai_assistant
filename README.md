# Conversational AI Assistant with Long-Term Memory

This project provides a friendly, precise, and context-aware AI assistant that leverages **vector databases for long-term memory**. The assistant can recall relevant information from all previous user conversations, making responses more accurate, contextually tailored, and useful.

---

## Features

- **Long-Term Memory:** All user interactions are persistently stored in a JSON file (and, in the future, can be stored in SQL/NoSQL databases for greater scalability). Whenever needed, this data is embedded and loaded into a vector database (Chroma) to enable efficient, context-aware retrieval of relevant memories across sessions.
- **Relevant Recall:** Retrieves only contextually related information from memory, improving the quality of responses.
- **Concise & Direct:** Answers queries with minimal filler, always aiming for usefulness and clarity.
- **Streaming & Traditional Responses:** Supports both single-turn and streaming LLM response modes.
- **Configurable Models:** Easily swap out LLM or embedding models by modifying environment variables.
- **Environment-Based Configuration:** All credentials and model/database names are securely managed via `.env` variables.
- **Startup Initialization:** The assistant auto-initializes its memory database for a fresh, ready-to-use environment.

---
## How It Works

1. **User Query:** The assistant receives a user question.
2. **Search Query Generation:** It uses an LLM to generate search terms related to the query.
3. **Memory Recall:** The vector database (`Chroma`) is searched for relevant past conversation snippets.
4. **Contextual Response:** The LLM answers the new prompt using the fetched context, always focusing on relevance and conciseness.

---

## Getting Started

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) (for local LLMs) or compatible LLM endpoints
- [Chroma](https://www.trychroma.com/) (for local vector database)
- [langchain](https://python.langchain.com/) and compatible LLM integrations (`langchain_chroma`, `langchain_ollama`)
- `python-dotenv` for environment management

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/ousaro/conversational_ai_assistant.git
    cd conversational_ai_assistant
    ```
2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Set up your `.env` configuration:**
    ```env
    # .env example
    EMBEDDING_MODEL=your-embedding-model-name
    CHAT_MODEL=your-chat-llm-model-name
    DATABASE_LOCATION=path/to/database_directory
    COLLECTION_NAME=your_collection_name
    ```

### Run The Assistant

```bash
python main.py
```

On first run, the vector database is (re)initialized, ensuring long-term memory is established and fresh.

---

## Code Overview

- `main.py`  
  Entry point; initializes the vector database and runs the assistant loop.
- `env.py`  
  Loads environment variables and ensures required parameters are set.
- `assistant.py`  
  Contains logic for the assistant interface and conversation handling.
- `prompt_utils.py`  
  Defines prompt templates for query generation and context formatting.
- `llm_utils.py`  
  Abstractions for invoking the LLM and supporting both single and streamed responses.
- `vector_db_utils.py`  
  Vector database initialization, reset, and management (for long-term memory support).

---

## Long-Term Memory (Vector Database Explanation)

The assistant **persistently records all conversations** in a primary data store (such as a JSON file today, or potentially an SQL/NoSQL database in the future). To power intelligent context recall, every conversation message is **embedded as a high-dimensional vector** and indexed in a fast, specialized vector database (Chroma). This architecture allows the system to:

- **Efficiently search for and recall only the most relevant snippets of past conversation** in real time, powering highly contextual responses.
- **Maintain context and memory across indefinite sessions and restarts** by loading and embedding records from persistent storage into the vector database on startup or update.
- **Avoid privacy leaks by restricting recall to only contextually linked previous interactions**, ensuring that retrieved memories are always relevant to the user’s current query.

This means the AI can tailor each response with just the right amount of prior, relevant information—making conversations feel consistent, intelligent, and highly personal, even over long periods or after restarts.

---

## Customization

- **Swap LLM models**  
  Change `CHAT_MODEL` in your `.env` to use a different large language model.
- **Modify embeddings**  
  Update `EMBEDDING_MODEL` for your preferred embedding generator.
- **Collection/database path**  
  Adjust `DATABASE_LOCATION` and `COLLECTION_NAME` as needed for safe, persistent storage.

---

**Note:**  
This long-term memory system is still being improved and tuned. Because of limited resources and current LLM limitations, things like recall accuracy and handling large amounts of data are still experimental. More work is needed to make these features fully reliable.

---

**[License](LICENSE):** MIT
