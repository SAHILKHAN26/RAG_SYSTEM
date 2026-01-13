# Bot Case Study - System Overview

## 🏗️ Architecture

The system is split into two main scopes: **Global Chat (Orchestrator)** and **Conversation APIs (Persistent Chat)**.

### 1. Global Chat (Orchestrator) 🌐
**Scope**: Stateless, routing-based interaction. The Orchestrator decides which agent handles your query.
-   **URL**: `http://localhost:8001`
-   **Method**: `POST /` (JSON-RPC)
-   **Behavior**: Analyzes request -> Routes to `GeneralAgent` or `RAGAgent` -> Returns response.
-   **Use Case**: "One-stop shop" for any query.

**Agents:**
*   **GeneralAgent** (Port `10003`): Handles open chat, greetings, asking for advice.
*   **RAGAgent** (Port `10002`): Handles questions about uploaded documents (Auto RAG).

### 2. Conversation APIs (REST) 💾
**Scope**: Stateful, persistent chat history stored in `bot_gpt.db`.
-   **URL**: `http://localhost:8000`
-   **Swagger UI**: `http://localhost:8000/docs`
-   **Behavior**: Manages users, uploads, and conversation history. Internally uses `RAGAgent` logic for responses.

---
## 🧰 Used Technologies

- **LangGraph** — Agents and workflows implemented with LangGraph (RAG and General agents).
- **ADK** — Agent Development Kit used for building and composing agents.
- **A2A** — Agent-to-Agent communication (Orchestrator routing between agents).

> Note: The project also uses **FAISS** (vector store), **FastAPI** (REST API), **SQLite** for persistence, and Python 3.11+.

---

## ⚙️ Steps to use

1. **Environment setup**

   ```bash
   # Create a virtual environment
   python -m venv .venv

   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1

   # macOS / Linux
   source .venv/bin/activate
   ```

2. **Install requirements**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**

   - Copy `.env example` to `.env` and update values (e.g., `DATABASE_URL`, `VECTOR_STORE_DIR`, `UPLOAD_DIR`).

---

## 🚀 Startup Guide

Run these commands in separate terminals:

### Step 1: Start Microservices (Agents)
```bash
# Terminal 1: RAG Agent (Port 10002)
python -m src.agents.langgraph.rag_agent

# Terminal 2: General Agent (Port 10003)
python -m src.agents.langgraph.general_agent
```

### Step 2: Start Global Chat (Orchestrator)
```bash
# Terminal 3: Orchestrator (Port 8001)
python host/entry.py
```

### Step 3: Start REST APIs (Main Server)
```bash
# Terminal 4: Main API (Port 8000)
python main.py
```





## 🔌 API Endpoints (High Level)

### Global Chat (Orchestrator)
-   **`POST /`**: Send a task/query to the intelligent router.
    ```json
    {
      "method": "tasks/send",
      "params": {
        "id": "1",
        "sessionId": "session-123",
        "message": { "role": "user", "parts": [{ "text": "Hello" }] }
      }
    }
    ```

### Conversation REST APIs (`http://localhost:8000`)
**Conversation Management**
-   `POST /api/v1/conversations`: Create a new chat session.
-   `POST /api/v1/conversations/{id}/messages`: Send a message to an existing chat.
-   `GET /api/v1/conversations/{id}`: Get full chat history.

**Document Management**
-   `POST /api/v1/documents/upload`: Upload files (PDF, TXT, etc.) for RAG.
-   `GET /api/v1/documents`: List uploaded files.

**Users**
-   `POST /api/v1/users`: Create a new user.
