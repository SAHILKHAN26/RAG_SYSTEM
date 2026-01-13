"""
LangGraph-based RAG Agent for conversational AI with document retrieval.

This module implements a LangGraph workflow with:
- Router node: Determines if retrieval is needed
- Retrieval node: Fetches relevant chunks from vector store
- Generation node: Generates response using LLM
- History management: Loads and saves conversation history
"""

from typing import TypedDict, List, Optional, Annotated
from operator import add

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import Conversation, Message, MessageRole, ConversationMode
from src.core.services.vector_store import VectorStore, get_vector_store
from src.core.services.llm_service import LLMService, LLMServiceFactory
from src.core.services.conversation_manager import ConversationManager, get_conversation_manager
from src.core.logger import app_logger as logger


# ============================================================================
# State Definition
# ============================================================================

class RAGState(TypedDict):
    """State for the RAG agent workflow"""
    conversation_id: str
    user_message: str
    chat_history: Annotated[List[BaseMessage], add]
    retrieved_context: Optional[List[str]]
    generated_response: str
    mode: str  # 'open_chat' or 'rag'
    document_ids: List[str]
    total_tokens: int
    error: Optional[str]


# ============================================================================
# RAG Agent
# ============================================================================

class RAGAgent:
    """LangGraph-based RAG agent for conversational AI"""
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        llm_service: Optional[LLMService] = None,
        conversation_manager: Optional[ConversationManager] = None,
    ):
        """
        Initialize the RAG agent.
        
        Args:
            vector_store: Vector store for document retrieval
            llm_service: LLM service for generation
            conversation_manager: Conversation manager for history
        """
        self.vector_store = vector_store or get_vector_store()
        self.llm_service = llm_service or LLMServiceFactory.get_service()
        self.conversation_manager = conversation_manager or get_conversation_manager(self.llm_service)
        
        # Build the LangGraph workflow
        self.graph = self._build_graph()
        
        logger.info("RAG agent initialized")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        # Create the graph
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("retrieve", self._retrieval_node)
        workflow.add_node("generate", self._generation_node)
        
        # Add edges
        workflow.set_entry_point("router")
        
        # Router decides: retrieve or generate directly
        workflow.add_conditional_edges(
            "router",
            self._should_retrieve,
            {
                "retrieve": "retrieve",
                "generate": "generate",
            }
        )
        
        # After retrieval, go to generation
        workflow.add_edge("retrieve", "generate")
        
        # After generation, end
        workflow.add_edge("generate", END)
        
        return workflow
    
    async def _router_node(self, state: RAGState) -> RAGState:
        """
        Router node: Determines if retrieval is needed.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        logger.info(f"Router node: mode={state['mode']}")
        
        # If mode is RAG and we have documents, we need retrieval
        if state["mode"] == ConversationMode.RAG.value and state.get("document_ids"):
            logger.info("Routing to retrieval")
        else:
            logger.info("Routing to direct generation")
        
        return state
    
    def _should_retrieve(self, state: RAGState) -> str:
        """
        Conditional edge: Decide if retrieval is needed.
        
        Args:
            state: Current state
            
        Returns:
            Next node name
        """
        if state["mode"] == ConversationMode.RAG.value and state.get("document_ids"):
            return "retrieve"
        return "generate"
    
    async def _retrieval_node(self, state: RAGState) -> RAGState:
        """
        Retrieval node: Fetch relevant chunks from vector store.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with retrieved context
        """
        logger.info("Retrieval node: fetching relevant chunks")
        
        try:
            query = state["user_message"]
            document_ids = state.get("document_ids", [])
            
            # Search for relevant chunks across all documents in a single search
            if document_ids:
                # Pass list of document IDs to filter - vector store will search all at once
                top_results = self.vector_store.search(
                    query=query,
                    top_k=3,  # Get top 3 chunks across all documents
                    filter_dict={"document_id": document_ids}  # List of document IDs
                )
                logger.info(f"Retrieved top 3 chunks from {len(document_ids)} documents in single search")
            else:
                # No specific documents, search all
                top_results = self.vector_store.search(query=query, top_k=3)
                logger.info("Retrieved top 3 chunks from all documents")
            
            # Extract text from results
            retrieved_texts = [result[0] for result in top_results]
            
            state["retrieved_context"] = retrieved_texts
            logger.info(f"Retrieved {len(retrieved_texts)} chunks with best relevance scores")
        
        except Exception as e:
            logger.error(f"Error in retrieval node: {e}", exc_info=True)
            state["retrieved_context"] = []
            state["error"] = f"Retrieval error: {str(e)}"
        
        return state
    
    async def _generation_node(self, state: RAGState) -> RAGState:
        """
        Generation node: Generate response using LLM.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with generated response
        """
        logger.info("Generation node: generating response")
        
        try:
            # Prepare messages
            messages = []
            
            # Add retrieved context if available
            if state.get("retrieved_context"):
                context_text = "\n\n".join([
                    f"[Context {i+1}]\n{chunk}"
                    for i, chunk in enumerate(state["retrieved_context"])
                ])
                
                system_prompt = f"""You are a helpful AI assistant. Answer the user's question based on the following context.

Context:
{context_text}

Instructions:
- Use ONLY the information from the provided context
- If the context doesn't contain enough information, say so
- Be specific and cite relevant parts of the context
"""
                messages.append({"role": "system", "content": system_prompt})
            
            # Add chat history
            for msg in state.get("chat_history", []):
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages.append({"role": "assistant", "content": msg.content})
            
            # Add current user message
            messages.append({"role": "user", "content": state["user_message"]})
            
            # Convert to LangChain messages
            from src.core.services.llm_service import create_user_message, create_assistant_message, create_system_message
            
            lc_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    lc_messages.append(create_system_message(msg["content"]))
                elif msg["role"] == "user":
                    lc_messages.append(create_user_message(msg["content"]))
                elif msg["role"] == "assistant":
                    lc_messages.append(create_assistant_message(msg["content"]))
            
            # Generate response
            response = await self.llm_service.generate(lc_messages)
            
            state["generated_response"] = response
            
            # Count tokens
            state["total_tokens"] = self.llm_service.count_message_tokens(lc_messages)
            state["total_tokens"] += self.llm_service.count_tokens(response)
            
            logger.info(f"Generated response: {len(response)} chars, ~{state['total_tokens']} tokens")
        
        except Exception as e:
            logger.error(f"Error in generation node: {e}", exc_info=True)
            state["generated_response"] = "I apologize, but I encountered an error generating a response. Please try again."
            state["error"] = f"Generation error: {str(e)}"
        
        return state
    
    async def process_message(
        self,
        session: AsyncSession,
        conversation_id: str,
        user_message: str,
        mode: str = ConversationMode.OPEN_CHAT.value,
        document_ids: Optional[List[str]] = None,
    ) -> tuple[str, int]:
        """
        Process a user message and generate a response.
        
        Args:
            session: Database session
            conversation_id: ID of the conversation
            user_message: User's message
            mode: Conversation mode ('open_chat' or 'rag')
            document_ids: Optional list of document IDs for RAG
            
        Returns:
            Tuple of (response, token_count)
        """
        logger.info(f"Processing message for conversation {conversation_id}")
        
        try:
            # Load conversation history
            messages, _ = await self.conversation_manager.prepare_context(
                session=session,
                conversation_id=conversation_id,
                new_message=user_message,
                retrieved_context=None,  # Will be filled by retrieval node
            )
            
            # Initialize state
            initial_state: RAGState = {
                "conversation_id": conversation_id,
                "user_message": user_message,
                "chat_history": messages[:-1],  # Exclude the new message
                "retrieved_context": None,
                "generated_response": "",
                "mode": mode,
                "document_ids": document_ids or [],
                "total_tokens": 0,
                "error": None,
            }
            
            # Compile and run the graph
            compiled_graph = self.graph.compile()
            
            # Run the workflow
            final_state = await compiled_graph.ainvoke(initial_state)
            
            response = final_state["generated_response"]
            token_count = final_state["total_tokens"]
            
            if final_state.get("error"):
                logger.warning(f"Workflow completed with error: {final_state['error']}")
            
            return response, token_count
        
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return "I apologize, but I encountered an error processing your message. Please try again.", 0


# Global RAG agent instance
_rag_agent_instance: Optional[RAGAgent] = None


def get_rag_agent() -> RAGAgent:
    """Get or create the global RAG agent instance"""
    global _rag_agent_instance
    if _rag_agent_instance is None:
        _rag_agent_instance = RAGAgent()
    return _rag_agent_instance
