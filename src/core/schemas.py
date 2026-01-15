"""
Pydantic schemas for request/response validation in BOT GPT API.

This module defines schemas for:
- User creation and responses
- Conversation CRUD operations
- Message handling
- Document uploads
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, EmailStr, validator
from enum import Enum


# Enums (matching database enums)
class ConversationMode(str, Enum):
    """Conversation mode"""
    OPEN_CHAT = "open_chat"
    RAG = "rag"


class MessageRole(str, Enum):
    """Message role"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class EmbeddingStatus(str, Enum):
    """Document embedding status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================
# User Schemas
# ============================================================================

class UserCreate(BaseModel):
    """Schema for creating a new user"""
    username: str = Field(..., min_length=3, max_length=100, description="Unique username")
    email: EmailStr = Field(..., description="User email address")


class UserResponse(BaseModel):
    """Schema for user response"""
    id: str
    username: str
    email: str
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============================================================================
# Message Schemas
# ============================================================================

class MessageBase(BaseModel):
    """Base message schema"""
    role: MessageRole
    content: str = Field(..., min_length=1, description="Message content")


class MessageCreate(BaseModel):
    """Schema for creating a message"""
    content: str = Field(..., min_length=1, max_length=10000, description="Message content")


class MessageResponse(BaseModel):
    """Schema for message response"""
    id: str
    conversation_id: str
    role: MessageRole
    content: str
    token_count: int
    sequence_number: int
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============================================================================
# Document Schemas
# ============================================================================

class DocumentUploadResponse(BaseModel):
    """Schema for document upload response"""
    id: str
    filename: str
    file_type: str
    file_size: int
    uploaded_at: datetime
    embedding_status: EmbeddingStatus
    
    class Config:
        from_attributes = True


class DocumentResponse(BaseModel):
    """Schema for document details"""
    id: str
    user_id: str
    filename: str
    file_type: str
    file_size: int
    uploaded_at: datetime
    chunk_count: int
    embedding_status: EmbeddingStatus
    error_message: Optional[str] = None
    
    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """Schema for paginated document list"""
    documents: List[DocumentResponse]
    total: int
    limit: int
    offset: int


# ============================================================================
# Conversation Schemas
# ============================================================================

class ConversationCreate(BaseModel):
    """Schema for creating a new conversation"""
    user_id: str = Field(..., description="User ID who owns the conversation")
    title: str = Field(..., min_length=1, max_length=255, description="Conversation title")
    mode: ConversationMode = Field(default=ConversationMode.OPEN_CHAT, description="Conversation mode (per message)")
    first_message: str = Field(..., min_length=1, max_length=10000, description="First message in the conversation")
    document_ids: Optional[List[str]] = Field(default=None, description="Document IDs for RAG mode")
    conversation_id: str = Field(..., description="Conversation ID who owns the conversation")


class ConversationUpdate(BaseModel):
    """Schema for updating conversation metadata"""
    title: Optional[str] = Field(None, min_length=1, max_length=255)


class ConversationSummary(BaseModel):
    """Schema for conversation summary (list view)"""
    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    total_tokens: int
    message_count: int = Field(default=0, description="Number of messages in conversation")
    
    class Config:
        from_attributes = True


class ConversationResponse(BaseModel):
    """Schema for full conversation details"""
    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    total_tokens: int
    messages: List[MessageResponse] = Field(default_factory=list)
    documents: List[DocumentResponse] = Field(default_factory=list)
    
    class Config:
        from_attributes = True


class ConversationListResponse(BaseModel):
    """Schema for paginated conversation list"""
    conversations: List[ConversationSummary]
    total: int
    limit: int
    offset: int


# ============================================================================
# Message Addition Schemas
# ============================================================================

class AddMessageRequest(BaseModel):
    """Schema for adding a message to existing conversation"""
    conversation_id: str = Field(..., description="Conversation ID")
    content: str = Field(..., min_length=1, max_length=10000, description="Message content")


class AddMessageResponse(BaseModel):
    """Schema for message addition response"""
    conversation_id: str
    user_message: MessageResponse
    assistant_message: MessageResponse
    
    class Config:
        from_attributes = True


class ConversationMessageResponse(BaseModel):
    """Schema for conversation response with only last message"""
    conversation_id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    total_tokens: int
    last_user_message: MessageResponse
    last_assistant_message: MessageResponse
    
    class Config:
        from_attributes = True


# ============================================================================
# Error Schemas
# ============================================================================

class ErrorResponse(BaseModel):
    """Schema for error responses"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict] = Field(None, description="Additional error details")


# ============================================================================
# Pagination Schemas
# ============================================================================

class PaginationParams(BaseModel):
    """Schema for pagination parameters"""
    limit: int = Field(default=20, ge=1, le=100, description="Number of items per page")
    offset: int = Field(default=0, ge=0, description="Number of items to skip")


# ============================================================================
# Chat Streaming Schemas
# ============================================================================

class StreamChunk(BaseModel):
    """Schema for streaming response chunks"""
    type: str = Field(..., description="Chunk type: 'token', 'metadata', 'error'")
    content: str = Field(default="", description="Chunk content")
    metadata: Optional[dict] = Field(None, description="Additional metadata")


# ============================================================================
# Statistics Schemas
# ============================================================================

class ConversationStats(BaseModel):
    """Schema for conversation statistics"""
    total_conversations: int
    total_messages: int
    total_tokens: int
    conversations_by_mode: dict = Field(default_factory=dict)
    avg_messages_per_conversation: float
    avg_tokens_per_conversation: float
