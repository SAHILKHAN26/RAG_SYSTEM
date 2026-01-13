# BOT GPT - Conversational AI Backend with RAG

A production-grade conversational AI platform with **Retrieval-Augmented Generation (RAG)** capabilities, built using **FastAPI**, **LangGraph**, **SQLite**, and **FAISS**.

## 🚀 Features

### Core Capabilities
- ✅ **Open Chat Mode**: General conversational AI without document context
- ✅ **RAG Mode**: Document-grounded conversations with semantic search
- ✅ **Conversation History**: Persistent storage with SQLite
- ✅ **Multi-LLM Support**: OpenAI (GPT-4, GPT-3.5) and Google Gemini
- ✅ **Vector Search**: FAISS for fast semantic document retrieval
- ✅ **LangGraph Workflow**: Intelligent routing between retrieval and generation
- ✅ **Token Management**: Sliding window and context optimization
- ✅ **Document Processing**: PDF, TXT, DOCX, MD support

### API Endpoints
- User management (create, retrieve)
- Conversation CRUD (create, list, get, delete)
- Message handling (add messages to conversations)
- Document upload and management
- Full REST API with pagination

## 📋 Requirements

- Python 3.10+
- OpenAI API key (for GPT models and embeddings)
- Google API key (optional, for Gemini models)

## 🛠️ Installation

### 1. Clone the repository
```bash
cd bot_case_study
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file in the project root:

```bash
# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
ENV=development

# API Keys
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Database
DATABASE_URL=sqlite+aiosqlite:///./bot_gpt.db

# Vector Store (FAISS)
VECTOR_STORE_DIR=./faiss_index
EMBEDDING_MODEL=text-embedding-3-small

# RAG Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=3

# Context Management
MAX_HISTORY_TOKENS=4000
MAX_CONTEXT_TOKENS=8000
SLIDING_WINDOW_SIZE=10

# LLM Settings
DEFAULT_LLM_MODEL=gpt-4
DEFAULT_TEMPERATURE=0.7
MAX_RETRIES=3

# File Upload
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=10485760
```

### 4. Initialize the database
```bash
python scripts/init_db.py --drop --seed
```

This will:
- Create all database tables
- Seed sample data (demo user)

## 🚀 Running the Application

### Start the server
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Access API documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📚 API Usage Examples

### 1. Create a User
```bash
curl -X POST "http://localhost:8000/api/v1/users" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john_doe",
    "email": "john@example.com"
  }'
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "username": "john_doe",
  "email": "john@example.com",
  "created_at": "2026-01-13T14:00:00Z"
}
```

### 2. Create an Open Chat Conversation
```bash
curl -X POST "http://localhost:8000/api/v1/conversations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "title": "General Chat",
    "mode": "open_chat",
    "first_message": "Hello! How are you today?"
  }'
```

**Response:**
```json
{
  "id": "conv-123",
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "title": "General Chat",
  "mode": "open_chat",
  "created_at": "2026-01-13T14:00:00Z",
  "updated_at": "2026-01-13T14:00:00Z",
  "total_tokens": 150,
  "messages": [
    {
      "id": "msg-1",
      "role": "user",
      "content": "Hello! How are you today?",
      "created_at": "2026-01-13T14:00:00Z"
    },
    {
      "id": "msg-2",
      "role": "assistant",
      "content": "Hello! I'm doing well, thank you for asking...",
      "created_at": "2026-01-13T14:00:05Z"
    }
  ],
  "documents": []
}
```

### 3. Upload a Document
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@/path/to/document.pdf" \
  -F "user_id=550e8400-e29b-41d4-a716-446655440000"
```

**Response:**
```json
{
  "id": "doc-456",
  "filename": "document.pdf",
  "file_type": "pdf",
  "file_size": 102400,
  "uploaded_at": "2026-01-13T14:05:00Z",
  "embedding_status": "completed"
}
```

### 4. Create a RAG Conversation
```bash
curl -X POST "http://localhost:8000/api/v1/conversations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "title": "Document Q&A",
    "mode": "rag",
    "first_message": "What are the main topics discussed in this document?",
    "document_ids": ["doc-456"]
  }'
```

### 5. Add a Message to Existing Conversation
```bash
curl -X POST "http://localhost:8000/api/v1/conversations/conv-123/messages" \
  -F "content=Tell me more about that topic"
```

### 6. List Conversations
```bash
curl -X GET "http://localhost:8000/api/v1/conversations?user_id=550e8400-e29b-41d4-a716-446655440000&limit=10&offset=0"
```

### 7. Get Conversation Details
```bash
curl -X GET "http://localhost:8000/api/v1/conversations/conv-123"
```

### 8. Delete a Conversation
```bash
curl -X DELETE "http://localhost:8000/api/v1/conversations/conv-123"
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI Server                        │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Conversation │  │   Document   │  │     User     │      │
│  │   Routes     │  │   Routes     │  │   Routes     │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
│         └──────────────────┴──────────────────┘              │
│                            │                                 │
│                    ┌───────▼────────┐                        │
│                    │   RAG Agent    │                        │
│                    │  (LangGraph)   │                        │
│                    └───────┬────────┘                        │
│                            │                                 │
│         ┌──────────────────┼──────────────────┐              │
│         │                  │                  │              │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────┐      │
│  │ Conversation │  │    Vector    │  │     LLM      │      │
│  │   Manager    │  │    Store     │  │   Service    │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────┐      │
│  │   SQLite     │  │   ChromaDB   │  │ OpenAI/Gemini│      │
│  │   Database   │  │              │  │     APIs     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 LangGraph Workflow

The RAG agent uses a LangGraph workflow with the following nodes:

1. **Router Node**: Determines if document retrieval is needed based on conversation mode
2. **Retrieval Node**: Fetches relevant chunks from ChromaDB (only for RAG mode)
3. **Generation Node**: Generates response using LLM with context and history

```
┌─────────┐
│  Start  │
└────┬────┘
     │
┌────▼────┐
│ Router  │
└────┬────┘
     │
     ├─── RAG Mode ───┐
     │                │
     │           ┌────▼────────┐
     │           │  Retrieval  │
     │           └────┬────────┘
     │                │
     └─── Open Chat ──┤
                      │
                 ┌────▼────────┐
                 │ Generation  │
                 └────┬────────┘
                      │
                 ┌────▼────┐
                 │   End   │
                 └─────────┘
```

## 💾 Database Schema

### Users
- `id` (UUID, PK)
- `username` (String, unique)
- `email` (String, unique)
- `created_at` (DateTime)

### Conversations
- `id` (UUID, PK)
- `user_id` (UUID, FK)
- `title` (String)
- `mode` (Enum: open_chat, rag)
- `created_at`, `updated_at` (DateTime)
- `total_tokens` (Integer)

### Messages
- `id` (UUID, PK)
- `conversation_id` (UUID, FK)
- `role` (Enum: user, assistant, system)
- `content` (Text)
- `token_count` (Integer)
- `sequence_number` (Integer)
- `created_at` (DateTime)

### Documents
- `id` (UUID, PK)
- `user_id` (UUID, FK)
- `filename` (String)
- `file_path` (String)
- `file_type` (String)
- `file_size` (Integer)
- `chunk_count` (Integer)
- `embedding_status` (Enum: pending, processing, completed, failed)
- `uploaded_at` (DateTime)

## 🎯 Token Management

The system implements intelligent token management:

1. **Sliding Window**: Keeps last N messages in context
2. **Token Counting**: Tracks tokens per message and conversation
3. **Context Optimization**: Fits history within token budget
4. **Priority System**:
   - Current user message (always included)
   - Last 5-10 messages (recent context)
   - Retrieved RAG context (top 3 chunks)
   - Total budget: ~8000 tokens for context

## 🔍 RAG Strategy

### Document Processing
1. **Upload**: User uploads PDF/TXT/DOCX/MD file
2. **Extraction**: Text extracted using appropriate parser
3. **Chunking**: Text split into 1000-char chunks with 200-char overlap
4. **Embedding**: Chunks embedded using OpenAI `text-embedding-3-small`
5. **Storage**: Embeddings stored in ChromaDB

### Retrieval
1. **Query**: User asks a question
2. **Search**: Semantic search in ChromaDB (top 3 chunks)
3. **Context**: Retrieved chunks added to LLM prompt
4. **Generation**: LLM generates response grounded in context

## 🧪 Testing

### Run the initialization script
```bash
python scripts/init_db.py --drop --seed
```

### Test the API
Use the Swagger UI at `http://localhost:8000/docs` to test all endpoints interactively.

## 📈 Scalability Considerations

### Current Setup (Development)
- SQLite for conversation storage
- ChromaDB for vector search
- Single-instance deployment

### Production Recommendations
1. **Database**: Migrate to PostgreSQL with connection pooling
2. **Vector Store**: Use Pinecone/Weaviate for distributed search
3. **Caching**: Add Redis for conversation history caching
4. **Queue**: Use Celery for async document processing
5. **Storage**: Use S3/GCS for document files
6. **Scaling**: Deploy multiple API instances behind load balancer

## 🔐 Security Notes

- API keys should be stored in environment variables
- File uploads are validated for type and size
- Database uses parameterized queries (SQLAlchemy ORM)
- CORS is enabled for development (configure for production)

## 📝 License

See LICENSE file for details.

## 🤝 Contributing

This is a case study implementation. For production use, consider:
- Adding authentication/authorization
- Implementing rate limiting
- Adding comprehensive error handling
- Setting up monitoring and logging
- Writing comprehensive tests

## 📧 Support

For questions or issues, please refer to the implementation plan document.
