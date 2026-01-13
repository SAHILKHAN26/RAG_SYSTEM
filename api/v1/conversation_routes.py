"""
API routes for conversation management.

Endpoints:
- POST /conversations - Create new conversation
- GET /conversations - List conversations
- GET /conversations/{id} - Get conversation details
- POST /conversations/{id}/messages - Add message to conversation
- DELETE /conversations/{id} - Delete conversation
- POST /users - Create user
- GET /users/{id} - Get user details
- POST /documents/upload - Upload document
- GET /documents - List documents
- DELETE /documents/{id} - Delete document
"""

import os
import uuid
from datetime import datetime
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.database import (
    get_db,
    User,
    Conversation,
    Message,
    Document,
    MessageRole,
    ConversationMode,
    EmbeddingStatus,
    conversation_document,
)
from src.core.schemas import (
    UserCreate,
    UserResponse,
    ConversationCreate,
    ConversationResponse,
    ConversationSummary,
    ConversationListResponse,
    AddMessageRequest,
    AddMessageResponse,
    MessageResponse,
    DocumentResponse,
    DocumentListResponse,
    DocumentUploadResponse,
    PaginationParams,
)
from src.core.services.vector_store import get_vector_store
from src.core.services.document_processor import get_document_processor
from src.agents.langgraph.rag_agent import get_rag_agent
from src.core.constant import UPLOAD_DIR, MAX_FILE_SIZE, ALLOWED_FILE_TYPES
from src.core.logger import app_logger as logger


# Create routers
conversation_router = APIRouter()
user_router = APIRouter()
document_router = APIRouter()


# ============================================================================
# User Endpoints
# ============================================================================

@user_router.post("/users", response_model=UserResponse, status_code=201)
async def create_user(
    user_data: UserCreate,
    session: AsyncSession = Depends(get_db),
):
    """Create a new user"""
    try:
        # Check if username or email already exists
        existing = await session.execute(
            select(User).where(
                (User.username == user_data.username) | (User.email == user_data.email)
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Username or email already exists")
        
        # Create user
        user = User(
            username=user_data.username,
            email=user_data.email,
        )
        
        session.add(user)
        await session.commit()
        await session.refresh(user)
        
        logger.info(f"Created user: {user.id}")
        return user
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user: {e}", exc_info=True)
        await session.rollback()
        raise HTTPException(status_code=500, detail="Failed to create user")


@user_router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    session: AsyncSession = Depends(get_db),
):
    """Get user details"""
    try:
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return user
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get user")


# ============================================================================
# Conversation Endpoints
# ============================================================================

@conversation_router.post("/conversations", response_model=ConversationResponse, status_code=201)
async def create_conversation(
    conv_data: ConversationCreate,
    session: AsyncSession = Depends(get_db),
):
    """Create a new conversation with first message"""
    try:
        # Verify user exists
        user_result = await session.execute(
            select(User).where(User.id == conv_data.user_id)
        )
        if not user_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="User not found")
        
        # Verify documents exist if provided
        if conv_data.document_ids:
            for doc_id in conv_data.document_ids:
                doc_result = await session.execute(
                    select(Document).where(Document.id == doc_id)
                )
                if not doc_result.scalar_one_or_none():
                    raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        # Create conversation
        conversation = Conversation(
            user_id=conv_data.user_id,
            title=conv_data.title,
            mode=conv_data.mode,
        )
        
        session.add(conversation)
        await session.flush()  # Get the conversation ID
        
        # Associate documents if provided
        if conv_data.document_ids:
            for doc_id in conv_data.document_ids:
                await session.execute(
                    conversation_document.insert().values(
                        conversation_id=conversation.id,
                        document_id=doc_id,
                    )
                )
        
        # Add first user message
        user_msg = Message(
            conversation_id=conversation.id,
            role=MessageRole.USER,
            content=conv_data.first_message,
            sequence_number=0,
        )
        session.add(user_msg)
        await session.flush()
        
        # Generate response using RAG agent
        rag_agent = get_rag_agent()
        response_text, token_count = await rag_agent.process_message(
            session=session,
            conversation_id=conversation.id,
            user_message=conv_data.first_message,
            mode=conv_data.mode.value,
            document_ids=conv_data.document_ids,
        )
        
        # Add assistant message
        assistant_msg = Message(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content=response_text,
            token_count=token_count,
            sequence_number=1,
        )
        session.add(assistant_msg)
        
        # Update conversation token count
        conversation.total_tokens = token_count
        
        await session.commit()
        
        # Load full conversation with relationships
        await session.refresh(conversation)
        result = await session.execute(
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .options(selectinload(Conversation.documents))
            .where(Conversation.id == conversation.id)
        )
        full_conversation = result.scalar_one()
        
        logger.info(f"Created conversation: {conversation.id}")
        return full_conversation
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating conversation: {e}", exc_info=True)
        await session.rollback()
        raise HTTPException(status_code=500, detail="Failed to create conversation")


@conversation_router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    user_id: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    session: AsyncSession = Depends(get_db),
):
    """List conversations with pagination"""
    try:
        # Build query
        query = select(Conversation)
        count_query = select(func.count(Conversation.id))
        
        if user_id:
            query = query.where(Conversation.user_id == user_id)
            count_query = count_query.where(Conversation.user_id == user_id)
        
        # Get total count
        total_result = await session.execute(count_query)
        total = total_result.scalar()
        
        # Get conversations
        query = query.order_by(Conversation.created_at.desc()).limit(limit).offset(offset)
        result = await session.execute(query)
        conversations = result.scalars().all()
        
        # Get message counts for each conversation
        summaries = []
        for conv in conversations:
            msg_count_result = await session.execute(
                select(func.count(Message.id)).where(Message.conversation_id == conv.id)
            )
            msg_count = msg_count_result.scalar()
            
            summary = ConversationSummary(
                id=conv.id,
                user_id=conv.user_id,
                title=conv.title,
                mode=conv.mode,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
                total_tokens=conv.total_tokens,
                message_count=msg_count,
            )
            summaries.append(summary)
        
        return ConversationListResponse(
            conversations=summaries,
            total=total,
            limit=limit,
            offset=offset,
        )
    
    except Exception as e:
        logger.error(f"Error listing conversations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list conversations")


@conversation_router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    session: AsyncSession = Depends(get_db),
):
    """Get conversation details with full history"""
    try:
        result = await session.execute(
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .options(selectinload(Conversation.documents))
            .where(Conversation.id == conversation_id)
        )
        conversation = result.scalar_one_or_none()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return conversation
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get conversation")


@conversation_router.post("/conversations/{conversation_id}/messages", response_model=AddMessageResponse)
async def add_message(
    conversation_id: str,
    content: str = Form(...),
    session: AsyncSession = Depends(get_db),
):
    """Add a message to an existing conversation"""
    try:
        # Get conversation
        result = await session.execute(
            select(Conversation)
            .options(selectinload(Conversation.documents))
            .where(Conversation.id == conversation_id)
        )
        conversation = result.scalar_one_or_none()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get next sequence number
        seq_result = await session.execute(
            select(func.max(Message.sequence_number)).where(
                Message.conversation_id == conversation_id
            )
        )
        max_seq = seq_result.scalar() or -1
        next_seq = max_seq + 1
        
        # Add user message
        user_msg = Message(
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=content,
            sequence_number=next_seq,
        )
        session.add(user_msg)
        await session.flush()
        
        # Generate response
        rag_agent = get_rag_agent()
        document_ids = [doc.id for doc in conversation.documents]
        
        response_text, token_count = await rag_agent.process_message(
            session=session,
            conversation_id=conversation_id,
            user_message=content,
            mode=conversation.mode.value,
            document_ids=document_ids,
        )
        
        # Add assistant message
        assistant_msg = Message(
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=response_text,
            token_count=token_count,
            sequence_number=next_seq + 1,
        )
        session.add(assistant_msg)
        
        # Update conversation
        conversation.total_tokens += token_count
        conversation.updated_at = datetime.utcnow()
        
        await session.commit()
        await session.refresh(user_msg)
        await session.refresh(assistant_msg)
        
        logger.info(f"Added message to conversation: {conversation_id}")
        
        return AddMessageResponse(
            conversation_id=conversation_id,
            user_message=user_msg,
            assistant_message=assistant_msg,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding message: {e}", exc_info=True)
        await session.rollback()
        raise HTTPException(status_code=500, detail="Failed to add message")


@conversation_router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    session: AsyncSession = Depends(get_db),
):
    """Delete a conversation and all its messages"""
    try:
        result = await session.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = result.scalar_one_or_none()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        await session.delete(conversation)
        await session.commit()
        
        logger.info(f"Deleted conversation: {conversation_id}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}", exc_info=True)
        await session.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete conversation")


# ============================================================================
# Document Endpoints
# ============================================================================

@document_router.post("/documents/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    session: AsyncSession = Depends(get_db),
):
    """Upload a document for RAG"""
    try:
        # Verify user exists
        user_result = await session.execute(
            select(User).where(User.id == user_id)
        )
        if not user_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="User not found")
        
        # Validate file
        file_ext = Path(file.filename).suffix.lower().lstrip('.')
        if file_ext not in ALLOWED_FILE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {ALLOWED_FILE_TYPES}"
            )
        
        # Read file
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size ({file_size} bytes) exceeds maximum ({MAX_FILE_SIZE} bytes)"
            )
        
        # Create upload directory
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # Save file
        doc_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        # Create document record
        document = Document(
            id=doc_id,
            user_id=user_id,
            filename=file.filename,
            file_path=file_path,
            file_type=file_ext,
            file_size=file_size,
            embedding_status=EmbeddingStatus.PENDING,
        )
        
        session.add(document)
        await session.commit()
        await session.refresh(document)
        
        # Process document asynchronously (in background)
        # For now, process synchronously
        try:
            document.embedding_status = EmbeddingStatus.PROCESSING
            await session.commit()
            
            # Extract text
            doc_processor = get_document_processor()
            text, char_count = doc_processor.process_document(file_path, file_ext)
            
            # Chunk and embed
            vector_store = get_vector_store()
            chunks = vector_store.chunk_text(
                text=text,
                document_id=doc_id,
                metadata={
                    "filename": file.filename,
                    "file_type": file_ext,
                    "user_id": user_id,
                }
            )
            
            vector_store.add_document_chunks(chunks)
            
            # Update document
            document.chunk_count = len(chunks)
            document.embedding_status = EmbeddingStatus.COMPLETED
            await session.commit()
            
            logger.info(f"Processed document: {doc_id}, {len(chunks)} chunks")
        
        except Exception as e:
            logger.error(f"Error processing document: {e}", exc_info=True)
            document.embedding_status = EmbeddingStatus.FAILED
            document.error_message = str(e)
            await session.commit()
        
        await session.refresh(document)
        return document
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}", exc_info=True)
        await session.rollback()
        raise HTTPException(status_code=500, detail="Failed to upload document")


@document_router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    user_id: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    session: AsyncSession = Depends(get_db),
):
    """List documents with pagination"""
    try:
        query = select(Document)
        count_query = select(func.count(Document.id))
        
        if user_id:
            query = query.where(Document.user_id == user_id)
            count_query = count_query.where(Document.user_id == user_id)
        
        # Get total count
        total_result = await session.execute(count_query)
        total = total_result.scalar()
        
        # Get documents
        query = query.order_by(Document.uploaded_at.desc()).limit(limit).offset(offset)
        result = await session.execute(query)
        documents = result.scalars().all()
        
        return DocumentListResponse(
            documents=list(documents),
            total=total,
            limit=limit,
            offset=offset,
        )
    
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list documents")


@document_router.delete("/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    session: AsyncSession = Depends(get_db),
):
    """Delete a document and its embeddings"""
    try:
        result = await session.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete from vector store
        try:
            vector_store = get_vector_store()
            vector_store.delete_document(document_id)
        except Exception as e:
            logger.warning(f"Error deleting from vector store: {e}")
        
        # Delete file
        try:
            if os.path.exists(document.file_path):
                os.remove(document.file_path)
        except Exception as e:
            logger.warning(f"Error deleting file: {e}")
        
        # Delete from database
        await session.delete(document)
        await session.commit()
        
        logger.info(f"Deleted document: {document_id}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}", exc_info=True)
        await session.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete document")
