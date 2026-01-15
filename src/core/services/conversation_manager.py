"""
Conversation Manager for handling conversation history and context management.

Features:
- Load conversation history from database
- Sliding window for context management
- Token counting and optimization
- Message summarization
"""

from typing import List, Optional, Tuple
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from src.core.database import Conversation, Message, MessageRole
from src.core.constant import (
    MAX_HISTORY_TOKENS,
    MAX_CONTEXT_TOKENS,
    SLIDING_WINDOW_SIZE,
    CONVERSATION_HISTORY_LIMIT,
)
from src.core.services.llm_service import LLMService, create_user_message, create_assistant_message
from src.core.logger import app_logger as logger


class ConversationManager:
    """Manages conversation history and context optimization"""
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        max_history_tokens: int = MAX_HISTORY_TOKENS,
        sliding_window_size: int = SLIDING_WINDOW_SIZE,
    ):
        """
        Initialize the conversation manager.
        
        Args:
            llm_service: LLM service for token counting
            max_history_tokens: Maximum tokens to include in history
            sliding_window_size: Number of recent messages to keep
        """
        self.llm_service = llm_service
        self.max_history_tokens = max_history_tokens
        self.sliding_window_size = sliding_window_size
        logger.info(f"Conversation manager initialized with window size: {sliding_window_size}")
    
    async def load_conversation_history(
        self,
        session: AsyncSession,
        conversation_id: str,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """
        Load conversation history from database.
        
        Args:
            session: Database session
            conversation_id: ID of the conversation
            limit: Optional limit on number of messages
            
        Returns:
            List of Message objects ordered by sequence number
        """
        try:
            query = (
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.sequence_number.asc())
            )
            
            if limit:
                query = query.limit(limit)
            
            result = await session.execute(query)
            messages = result.scalars().all()
            
            logger.info(f"Loaded {len(messages)} messages for conversation {conversation_id}")
            return list(messages)
        
        except Exception as e:
            logger.error(f"Error loading conversation history: {e}", exc_info=True)
            raise
    
    async def get_message_count(
        self,
        session: AsyncSession,
        conversation_id: str,
    ) -> int:
        """
        Get the count of messages in a conversation.
        
        Args:
            session: Database session
            conversation_id: ID of the conversation
            
        Returns:
            Number of messages
        """
        try:
            query = select(func.count(Message.id)).where(
                Message.conversation_id == conversation_id
            )
            result = await session.execute(query)
            count = result.scalar()
            return count or 0
        
        except Exception as e:
            logger.error(f"Error getting message count: {e}", exc_info=True)
            return 0
    
    async def load_last_n_messages(
        self,
        session: AsyncSession,
        conversation_id: str,
        n: int = 5,
    ) -> List[Message]:
        """
        Load the last N messages from a conversation.
        
        Args:
            session: Database session
            conversation_id: ID of the conversation
            n: Number of recent messages to fetch (default: 5)
            
        Returns:
            List of last N Message objects ordered by sequence number (oldest to newest)
        """
        try:
            # Skip loading for temporary conversation IDs
            if conversation_id.startswith("temp-"):
                logger.info(f"Skipping history load for temporary conversation: {conversation_id}")
                return []
            
            # Query to get last N messages ordered by sequence_number descending, then reverse
            query = (
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.sequence_number.desc())
                .limit(n)
            )
            
            result = await session.execute(query)
            messages = result.scalars().all()
            
            # Reverse to get chronological order (oldest to newest)
            messages_list = list(reversed(list(messages)))
            
            logger.info(f"Loaded last {len(messages_list)} messages for conversation {conversation_id}")
            logger.info(f"messages are: {[msg.content for msg in messages_list]}")
            return messages_list
        
        except Exception as e:
            logger.error(f"Error loading last N messages: {e}", exc_info=True)
            return []
    
    def messages_to_langchain(self, messages: List[Message]) -> List[BaseMessage]:
        """
        Convert database messages to LangChain message format.
        
        Args:
            messages: List of database Message objects
            
        Returns:
            List of LangChain BaseMessage objects
        """
        langchain_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.USER:
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == MessageRole.ASSISTANT:
                langchain_messages.append(AIMessage(content=msg.content))
            elif msg.role == MessageRole.SYSTEM:
                langchain_messages.append(SystemMessage(content=msg.content))
        
        return langchain_messages
    
    def apply_sliding_window(
        self,
        messages: List[BaseMessage],
        window_size: Optional[int] = None,
    ) -> List[BaseMessage]:
        """
        Apply sliding window to keep only recent messages.
        
        Args:
            messages: List of messages
            window_size: Size of the window (defaults to self.sliding_window_size)
            
        Returns:
            List of recent messages
        """
        if window_size is None:
            window_size = self.sliding_window_size
        
        if len(messages) <= window_size:
            return messages
        
        # Keep the last N messages
        windowed_messages = messages[-window_size:]
        logger.info(f"Applied sliding window: {len(messages)} -> {len(windowed_messages)} messages")
        
        return windowed_messages
    
    def optimize_context(
        self,
        messages: List[BaseMessage],
        max_tokens: Optional[int] = None,
    ) -> List[BaseMessage]:
        """
        Optimize context to fit within token budget.
        
        Args:
            messages: List of messages
            max_tokens: Maximum tokens allowed (defaults to self.max_history_tokens)
            
        Returns:
            Optimized list of messages
        """
        if max_tokens is None:
            max_tokens = self.max_history_tokens
        
        if not self.llm_service:
            # If no LLM service, just apply sliding window
            return self.apply_sliding_window(messages)
        
        # Count current tokens
        current_tokens = self.llm_service.count_message_tokens(messages)
        
        if current_tokens <= max_tokens:
            return messages
        
        # Apply sliding window until we fit within budget
        window_size = len(messages)
        while window_size > 1:
            window_size -= 1
            windowed_messages = messages[-window_size:]
            tokens = self.llm_service.count_message_tokens(windowed_messages)
            
            if tokens <= max_tokens:
                logger.info(f"Optimized context: {current_tokens} -> {tokens} tokens")
                return windowed_messages
        
        # If still too large, return just the last message
        logger.warning("Context still too large, returning only last message")
        return messages[-1:] if messages else []
    
    async def create_summary(
        self,
        messages: List[BaseMessage],
    ) -> str:
        """
        Create a summary of conversation history.
        
        Args:
            messages: List of messages to summarize
            
        Returns:
            Summary text
        """
        if not self.llm_service:
            return "Previous conversation history (summary not available)"
        
        # Create summarization prompt
        conversation_text = "\n".join([
            f"{msg.__class__.__name__}: {msg.content}"
            for msg in messages
        ])
        
        summary_prompt = f"""Please provide a concise summary of the following conversation, 
        highlighting the main topics discussed and key points:

        {conversation_text}

        Summary:"""
        
        try:
            summary = await self.llm_service.generate([
                HumanMessage(content=summary_prompt)
            ])
            logger.info("Created conversation summary")
            return summary
        
        except Exception as e:
            logger.error(f"Error creating summary: {e}", exc_info=True)
            return "Previous conversation history"
    
    async def prepare_context(
        self,
        session: AsyncSession,
        conversation_id: str,
        new_message: str,
        retrieved_context: Optional[List[str]] = None,
    ) -> Tuple[List[BaseMessage], int]:
        """
        Prepare the full context for LLM including history and RAG context.
        
        Args:
            session: Database session
            conversation_id: ID of the conversation
            new_message: New user message
            retrieved_context: Optional retrieved context chunks for RAG
            
        Returns:
            Tuple of (messages, total_tokens)
        """
        logger.info(f"[ConversationManager] Preparing context for conversation {conversation_id}")
        logger.debug(f"[ConversationManager] New message length: {len(new_message)} chars")
        logger.debug(f"[ConversationManager] Retrieved context chunks: {len(retrieved_context) if retrieved_context else 0}")
        
        # Load last N messages from conversation history (using configurable limit)
        db_messages = await self.load_last_n_messages(session, conversation_id, n=CONVERSATION_HISTORY_LIMIT)
        logger.info(f"[ConversationManager] Retrieved {len(db_messages)} messages from database")
        
        # Convert to LangChain format
        history_messages = self.messages_to_langchain(db_messages)
        logger.debug(f"[ConversationManager] Converted {len(history_messages)} messages to LangChain format")
        
        # Optimize history to fit token budget
        # Reserve tokens for RAG context and new message
        rag_context_tokens = 0
        if retrieved_context:
            rag_text = "\n\n".join(retrieved_context)
            rag_context_tokens = self.llm_service.count_tokens(rag_text) if self.llm_service else len(rag_text) // 4
            logger.debug(f"[ConversationManager] RAG context tokens: {rag_context_tokens}")
        
        new_message_tokens = self.llm_service.count_tokens(new_message) if self.llm_service else len(new_message) // 4
        logger.debug(f"[ConversationManager] New message tokens: {new_message_tokens}")
        
        # Calculate available tokens for history
        available_for_history = MAX_CONTEXT_TOKENS - rag_context_tokens - new_message_tokens - 500  # 500 buffer
        logger.debug(f"[ConversationManager] Available tokens for history: {available_for_history}")
        
        optimized_history = self.optimize_context(
            history_messages,
            max_tokens=max(available_for_history, 1000)  # At least 1000 tokens for history
        )
        logger.info(f"[ConversationManager] Optimized history: {len(history_messages)} -> {len(optimized_history)} messages")
        
        # Build final message list
        final_messages = []
        
        # Add RAG context as system message if available
        if retrieved_context:
            context_text = "\n\n".join([
                f"[Context {i+1}]\n{chunk}"
                for i, chunk in enumerate(retrieved_context)
            ])
            
            system_prompt = f"""You are a helpful AI assistant. Answer the user's question based on the following context.
            
Context:
{context_text}

Instructions:
- Use ONLY the information from the provided context
- If the context doesn't contain enough information, say so
- Be specific and cite relevant parts of the context
"""
            final_messages.append(SystemMessage(content=system_prompt))
            logger.debug(f"[ConversationManager] Added RAG system message with {len(retrieved_context)} context chunks")
        
        # Add optimized history
        final_messages.extend(optimized_history)
        logger.debug(f"[ConversationManager] Added {len(optimized_history)} history messages")
        
        # Add new user message
        final_messages.append(HumanMessage(content=new_message))
        logger.debug(f"[ConversationManager] Added new user message")
        
        # Calculate total tokens
        total_tokens = self.llm_service.count_message_tokens(final_messages) if self.llm_service else 0
        
        logger.info(f"[ConversationManager] Context prepared: {len(final_messages)} total messages, ~{total_tokens} tokens")
        logger.debug(f"[ConversationManager] Breakdown - System: {1 if retrieved_context else 0}, History: {len(optimized_history)}, New: 1")
        
        return final_messages, total_tokens


# Global conversation manager instance
_conversation_manager_instance: Optional[ConversationManager] = None


def get_conversation_manager(llm_service: Optional[LLMService] = None) -> ConversationManager:
    """Get or create the global conversation manager instance"""
    global _conversation_manager_instance
    if _conversation_manager_instance is None:
        _conversation_manager_instance = ConversationManager(llm_service=llm_service)
    return _conversation_manager_instance
