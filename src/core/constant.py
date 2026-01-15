import os
from dotenv import load_dotenv
load_dotenv()

ENV_MODE  = os.getenv("ENV")
SERVER_PORT = os.getenv("SERVER_PORT")
SERVER_HOST = os.getenv("SERVER_HOST")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

ORCHESTRATOR_HOST = os.getenv("ORCHESTRATOR_HOST")
ORCHESTRATOR_PORT = os.getenv("ORCHESTRATOR_PORT")

HOST_REGISTRY_FILE = "src/utils/agent_registry_host.json"


# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./bot_gpt.db")

# Vector Store Configuration (FAISS)
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "./faiss_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# RAG Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "3"))

# Context Management
MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", "4000"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "8000"))
SLIDING_WINDOW_SIZE = int(os.getenv("SLIDING_WINDOW_SIZE", "10"))
CONVERSATION_HISTORY_LIMIT = int(os.getenv("CONVERSATION_HISTORY_LIMIT", "5"))  # Number of recent messages to load

# LLM Settings
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# File Upload Settings
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB in bytes
ALLOWED_FILE_TYPES = ["pdf", "txt", "docx", "md"]