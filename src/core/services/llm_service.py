"""
LLM Service for unified interface to multiple LLM providers.

Supports:
- OpenAI (GPT-4, GPT-3.5-turbo)
- Google Gemini (gemini-pro, gemini-1.5-pro)
"""

import asyncio
from typing import List, Optional, AsyncIterator
from enum import Enum

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel

from src.core.constant import (
    OPENAI_API_KEY,
    GOOGLE_API_KEY,
    DEFAULT_LLM_MODEL,
    DEFAULT_TEMPERATURE,
    MAX_RETRIES,
)
from src.core.logger import app_logger as logger


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    GEMINI = "gemini"


class LLMService:
    """Unified service for interacting with multiple LLM providers"""
    
    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_retries: int = MAX_RETRIES,
    ):
        """
        Initialize the LLM service.
        
        Args:
            model_name: Name of the model to use
            temperature: Temperature for generation (0.0 to 1.0)
            max_retries: Maximum number of retries on failure
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        
        # Determine provider from model name
        self.provider = self._detect_provider(model_name)
        
        # Initialize the LLM
        self.llm = self._initialize_llm()
        
        logger.info(f"LLM service initialized: {self.provider.value} - {model_name}")
    
    def _detect_provider(self, model_name: str) -> LLMProvider:
        """Detect the provider from the model name"""
        model_lower = model_name.lower()
        
        if "gpt" in model_lower or "openai" in model_lower:
            return LLMProvider.OPENAI
        elif "gemini" in model_lower or "google" in model_lower:
            return LLMProvider.GEMINI
        else:
            # Default to OpenAI
            logger.warning(f"Unknown model {model_name}, defaulting to OpenAI")
            return LLMProvider.OPENAI
    
    def _initialize_llm(self) -> BaseChatModel:
        """Initialize the appropriate LLM based on provider"""
        if self.provider == LLMProvider.OPENAI:
            if not OPENAI_API_KEY:
                raise ValueError("OpenAI API key not found in environment variables")
            
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_retries=self.max_retries,
                openai_api_key=OPENAI_API_KEY,
            )
        
        elif self.provider == LLMProvider.GEMINI:
            if not GOOGLE_API_KEY:
                raise ValueError("Google API key not found in environment variables")
            
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                google_api_key=GOOGLE_API_KEY,
            )
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def generate(
        self,
        messages: List[BaseMessage],
        stream: bool = False,
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of messages in the conversation
            stream: Whether to stream the response
            
        Returns:
            Generated response text
        """
        try:
            if stream:
                # Streaming response
                full_response = ""
                async for chunk in self.stream(messages):
                    full_response += chunk
                return full_response
            else:
                # Non-streaming response
                response = await self.llm.ainvoke(messages)
                return response.content
        
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}", exc_info=True)
            raise
    
    async def stream(self, messages: List[BaseMessage]) -> AsyncIterator[str]:
        """
        Stream a response from the LLM.
        
        Args:
            messages: List of messages in the conversation
            
        Yields:
            Response chunks
        """
        try:
            async for chunk in self.llm.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
        
        except Exception as e:
            logger.error(f"Error streaming LLM response: {e}", exc_info=True)
            raise
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text (approximate).
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Approximate token count
        """
        # Simple approximation: ~4 characters per token
        # For production, use tiktoken for OpenAI or proper tokenizer
        return len(text) // 4
    
    def count_message_tokens(self, messages: List[BaseMessage]) -> int:
        """
        Count total tokens in a list of messages.
        
        Args:
            messages: List of messages
            
        Returns:
            Total token count
        """
        total = 0
        for message in messages:
            total += self.count_tokens(message.content)
            # Add overhead for message formatting (~4 tokens per message)
            total += 4
        return total


class LLMServiceFactory:
    """Factory for creating LLM service instances"""
    
    _instances: dict = {}
    
    @classmethod
    def get_service(
        cls,
        model_name: str = DEFAULT_LLM_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> LLMService:
        """
        Get or create an LLM service instance.
        
        Args:
            model_name: Name of the model
            temperature: Temperature for generation
            
        Returns:
            LLM service instance
        """
        key = f"{model_name}_{temperature}"
        
        if key not in cls._instances:
            cls._instances[key] = LLMService(
                model_name=model_name,
                temperature=temperature,
            )
        
        return cls._instances[key]
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached instances"""
        cls._instances.clear()


# Helper functions for creating messages
def create_system_message(content: str) -> SystemMessage:
    """Create a system message"""
    return SystemMessage(content=content)


def create_user_message(content: str) -> HumanMessage:
    """Create a user message"""
    return HumanMessage(content=content)


def create_assistant_message(content: str) -> AIMessage:
    """Create an assistant message"""
    return AIMessage(content=content)


# Default system prompts
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. You provide accurate, concise, and helpful responses to user queries."""

RAG_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context.

When answering:
1. Use ONLY the information from the provided context
2. If the context doesn't contain enough information to answer the question, say so
3. Be specific and cite relevant parts of the context
4. If the question is not related to the context, politely redirect the user

Context:
{context}
"""
