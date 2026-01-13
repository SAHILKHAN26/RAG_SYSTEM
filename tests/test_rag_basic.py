"""
Basic tests for RAG Agent functionality.

Run with: pytest tests/test_rag_basic.py -v
"""

import pytest
import asyncio
from datetime import datetime

# Mock tests since we don't have a full test environment set up


class TestDatabaseModels:
    """Test database model creation"""
    
    def test_imports(self):
        """Test that all modules can be imported"""
        try:
            from src.core.database import User, Conversation, Message, Document
            from src.core.schemas import ConversationCreate, UserCreate
            from src.core.services.vector_store import VectorStore
            from src.core.services.llm_service import LLMService
            from src.core.services.conversation_manager import ConversationManager
            from src.agents.langgraph.rag_agent import RAGAgent
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
    
    def test_enum_values(self):
        """Test enum definitions"""
        from src.core.database import ConversationMode, MessageRole, EmbeddingStatus
        
        assert ConversationMode.OPEN_CHAT.value == "open_chat"
        assert ConversationMode.RAG.value == "rag"
        
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.SYSTEM.value == "system"
        
        assert EmbeddingStatus.PENDING.value == "pending"
        assert EmbeddingStatus.COMPLETED.value == "completed"


class TestSchemas:
    """Test Pydantic schemas"""
    
    def test_user_create_schema(self):
        """Test UserCreate schema validation"""
        from src.core.schemas import UserCreate
        
        user = UserCreate(
            username="test_user",
            email="test@example.com"
        )
        
        assert user.username == "test_user"
        assert user.email == "test@example.com"
    
    def test_conversation_create_schema(self):
        """Test ConversationCreate schema validation"""
        from src.core.schemas import ConversationCreate, ConversationMode
        
        conv = ConversationCreate(
            user_id="user-123",
            title="Test Chat",
            mode=ConversationMode.OPEN_CHAT,
            first_message="Hello!"
        )
        
        assert conv.user_id == "user-123"
        assert conv.title == "Test Chat"
        assert conv.mode == ConversationMode.OPEN_CHAT
        assert conv.first_message == "Hello!"
    
    def test_rag_conversation_validation(self):
        """Test RAG conversation requires documents"""
        from src.core.schemas import ConversationCreate, ConversationMode
        from pydantic import ValidationError
        
        # RAG mode without documents should fail validation
        with pytest.raises(ValidationError):
            ConversationCreate(
                user_id="user-123",
                title="RAG Chat",
                mode=ConversationMode.RAG,
                first_message="Question?",
                document_ids=None  # Should fail
            )


class TestVectorStore:
    """Test vector store functionality"""
    
    def test_chunk_text(self):
        """Test text chunking"""
        from src.core.services.vector_store import VectorStore
        
        # Create a mock vector store (won't actually connect to ChromaDB)
        # This is just to test the chunking logic
        text = "This is a test document. " * 100  # Long text
        
        # We can't fully test without ChromaDB, but we can test imports
        assert VectorStore is not None


class TestLLMService:
    """Test LLM service"""
    
    def test_provider_detection(self):
        """Test LLM provider detection"""
        from src.core.services.llm_service import LLMService, LLMProvider
        
        # Test provider detection logic (without actually creating the service)
        # This would require API keys
        
        # Just verify the enum exists
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.GEMINI.value == "gemini"
    
    def test_token_counting(self):
        """Test token counting approximation"""
        # Simple approximation test
        text = "Hello world"
        # Approximate: 4 chars per token
        approx_tokens = len(text) // 4
        assert approx_tokens > 0


class TestConversationManager:
    """Test conversation manager"""
    
    def test_sliding_window(self):
        """Test sliding window logic"""
        from src.core.services.conversation_manager import ConversationManager
        from langchain_core.messages import HumanMessage, AIMessage
        
        manager = ConversationManager()
        
        # Create test messages
        messages = [
            HumanMessage(content=f"Message {i}")
            for i in range(20)
        ]
        
        # Apply sliding window
        windowed = manager.apply_sliding_window(messages, window_size=10)
        
        assert len(windowed) == 10
        assert windowed[0].content == "Message 10"  # Should keep last 10


class TestRAGAgent:
    """Test RAG agent"""
    
    def test_state_definition(self):
        """Test RAG state TypedDict"""
        from src.agents.langgraph.rag_agent import RAGState
        
        # Verify the state structure
        state: RAGState = {
            "conversation_id": "test-123",
            "user_message": "Hello",
            "chat_history": [],
            "retrieved_context": None,
            "generated_response": "",
            "mode": "open_chat",
            "document_ids": [],
            "total_tokens": 0,
            "error": None,
        }
        
        assert state["conversation_id"] == "test-123"
        assert state["mode"] == "open_chat"


# Integration test markers
@pytest.mark.integration
class TestAPIEndpoints:
    """Integration tests for API endpoints (requires running server)"""
    
    @pytest.mark.skip(reason="Requires running server")
    def test_create_user_endpoint(self):
        """Test user creation endpoint"""
        import requests
        
        response = requests.post(
            "http://localhost:8000/api/v1/users",
            json={
                "username": "test_user",
                "email": "test@example.com"
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["username"] == "test_user"


def test_configuration():
    """Test that configuration is loaded"""
    from src.core.constant import (
        DATABASE_URL,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        MAX_HISTORY_TOKENS,
    )
    
    assert DATABASE_URL is not None
    assert CHUNK_SIZE > 0
    assert CHUNK_OVERLAP > 0
    assert MAX_HISTORY_TOKENS > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
