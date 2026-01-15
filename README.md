# Bot Case Study - System Overview

## 🏗️ Architecture

The system is split into two main scopes: **Global Chat (By Agentic AI System)** and **Conversation APIs (Persistent Chat)**.

### 1. Global Chat (By Agentic AI System) 🌐
The central "brain" that routes your query without picking documents to the best-suited agent.
-   **Internal URL**: `http://localhost:8001` (JSON-RPC)
-   **Router Logic**: Analyzes request -> Routes to `GeneralAgent` (Open Chat) or `RAGAgent` (Document Q&A).
-   **Public Endpoint**: `POST /orchestrator/global-chat` (via Main API).

**Agents:**
*   **GeneralAgent** (Port `10003`): Handles open chat, greetings, asking for advice.
*   **RAGAgent** (Port `10002`): Handles questions about uploaded documents (Auto RAG).

### 2. Conversation APIs (REST) 💾
The user-facing gateway that manages state, persistence, and external access.
-   **URL**: `http://localhost:8000`
-   **Swagger UI**: `http://localhost:8000/docs`
-   **Key Features**:
    -   **Persistent Global Chat**: Stores history of orchestrator interactions.
    -   **Conversation Management**: Manages chat sessions, users, and uploads.
    -   **Data Persistence**: Uses `bot_gpt.db` (SQLite) to store all messages and conversations.

---

## � Context Relevance Power (Memory)

One of the most powerful features of this system is **Stateful Context Awareness**.

**How it works:**
Whenever you provide a `conversation_id` in your request (whether it's Global Chat or a specific Conversation endpoint), the system automatically:
1.  **Retrieves History:** Fetches the previous messages from the `messages` database table for that specific ID.
2.  **injects Context:** Feeds this history back into the LLM along with your new query.
3.  **Maintains Continuity:** This allows the AI to "remember" what you said earlier.

> **Example:**
> 1. User: "My name is Sahil." (ID: `chat-1`)
> 2. User: "What is my name?" (ID: `chat-1`) -> **AI Answer:** "Your name is Sahil."
>
> *This works seamlessly across Global Chat, RAG, and Open Chat modes!*

---

## �🧰 Tech Stack

*   **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Async Python Framework)
*   **Agentic Framework**: [LangGraph](https://langchain-ai.github.io/langgraph/) (Stateful multi-agent orchestration)
*   **Agent Development**: Google's ADK & A2A (Agent-to-Agent Protocol)
*   **Database**: SQLite (via `aiosqlite`) + SQLAlchemy (Async ORM)
*   **Vector Querying**: FAISS (for RAG document indexing)
*   **Language**: Python 3.11+

---

## ⚙️ Setup & Run

1. **Environment setup**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # Windows
   # source .venv/bin/activate   # Mac/Linux
   ```

2. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   - Copy `.env example` to `.env` and fill in values (DB URL, Vector Store path, etc.).

4. **🚀 Launch System**
   Simply run the start script. This will **initialize the database** first, then launch all services.
   ```bash
   run.bat
   ```

---

## 🔌 API Endpoints Guide

### 1. Global Chat (Persistent) 🌐
**Endpoint:** `POST /orchestrator/global-chat`
**Use Case:** The main entry point for users. The system intelligently decides whether to chat casually or search documents.

**Request Body:**
```json
{
  "user_id": "string",
  "query": "string",
  "target_agents": [
    "RAGAgent", "GeneralAgent"
  ],
  "toggle_all": false, //true or false
  "conversation_id": "string"
}
```

### 2. Conversation Management 💬
**Endpoint:** `POST /api/v1/conversations`
**Use Case:** A conversation with a specific mode or documents. Here you can select mode and the select documents in which you want to chat .

**Request Body:**
```json
{
  "user_id": "user_123",
  "title": "My Research Chat",
  "mode": "rag",                     // or "open_chat"
  "first_message": "string",
  "document_ids": ["doc_1", "doc_2"], // Optional: Attach specific docs
  "conversation_id": "string"
}
```

**Get History:**
`GET /api/v1/conversations/{conversation_id}`

**Get All History of User:**
`GET /api/v1/all-conversations`

**Delete History:**
`DELETE /api/v1/conversations/{conversation_id}`

### 3. Document Management 📄
**Endpoint:** `POST /api/v1/documents/upload`
**Use Case:** Upload files for the RAG agent to index and search.

**Request Form Data:**
-   `file`: (The file object, e.g., PDF/TXT)
-   `user_id`: "user_123"

**Get Documents of User:**
`GET /api/v1/documents`

**Delete Document:**
`DELETE /api/v1/documents/{document_id}`

### 4. User Management 👤
**Endpoint:** `POST /api/v1/users`
**Use Case:** Register new users.

**Request Body:**
```json
{
  "username": "sahilkhan",
  "email": "sahil@example.com"
}
```
**Get User details:**
`GET /api/v1/users/{user_id}`
