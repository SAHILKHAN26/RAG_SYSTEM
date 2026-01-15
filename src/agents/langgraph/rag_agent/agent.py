"""
LangGraph-based RAG Agent for conversational AI with document retrieval.

This module implements a LangGraph workflow with:
- Router node: Determines if retrieval is needed
- Retrieval node: Fetches relevant chunks from vector store
- Generation node: Generates response using LLM
- History management: Loads and saves conversation history
"""

from typing import TypedDict, List, Optional, Annotated, Any
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
    document_ids: List[str]
    total_tokens: int
    error: Optional[str]


# ============================================================================
# RAG Agent
# ============================================================================

class RAGAgent:
    """LangGraph-based RAG agent for conversational AI"""
    

    RELEVANCE_THRESHOLD = 1.2
    
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
        
        workflow.add_node("retrieve", self._retrieval_node)
        workflow.add_node("generate", self._generation_node)
        
        # Add edges
        workflow.set_entry_point("retrieve")

        # After retrieval, go to generation
        workflow.add_edge("retrieve", "generate")
        
        # After generation, end
        workflow.add_edge("generate", END)
        
        return workflow
    

    async def _retrieval_node(self, state: RAGState) -> RAGState:
        """
        Retrieval node: Fetch relevant chunks from vector store.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with retrieved context
        """
        logger.info("Retrieval node: searching for relevant chunks")
        
        try:
            query = state["user_message"]
            document_ids = state.get("document_ids", [])
            
            top_results = []
            
            if document_ids:
                # Explicit filtering if document IDs are provided
                top_results = self.vector_store.search(
                    query=query,
                    top_k=3,
                    filter_dict={"document_id": document_ids}
                )
                logger.info(f"Retrieved top 3 chunks from {len(document_ids)} documents")
            else:
                # Auto RAG: Search EVERYTHING
                top_results = self.vector_store.search(query=query, top_k=3)
                
                # # Check relevance for Auto RAG
                # if top_results:
                #     best_distance = top_results[0][2]  # L2 distance
                #     logger.info(f"Auto RAG: Best match distance = {best_distance}")
                    
                #     # If best match is too far, discard results (fallback to open chat)
                #     # Note: L2 distance is unbounded, roughly 0.0 (exact) to 2.0+ (far) for normalized vectors
                #     # Adjust RELEVANCE_THRESHOLD as needed.
                #     if best_distance > self.RELEVANCE_THRESHOLD:
                #         logger.info(f"Auto RAG: Results irrelevant (>{self.RELEVANCE_THRESHOLD}). Switching to Open Chat.")
                #         top_results = []
                # else:
                #     logger.info("Auto RAG: No results found.")
            
            # Extract text from results
            retrieved_texts = [result[0] for result in top_results]
            
            state["retrieved_context"] = retrieved_texts
            
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
            else:
                # No context (Open Chat fallback)
                system_prompt = "You are a helpful AI assistant. Answer the user's question to the best of your ability."
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
        user_message: str,  # Simplified interface for TaskManager
        session: Optional[AsyncSession] = None,
        conversation_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
    ) -> Any:  # Returns dict for TaskManager or tuple for internal use
        """
        Process a user message and generate a response.
        Compatible with both TaskManager (Orchestrator) and direct API use.
        """
        
        # If called from TaskManager (Orchestrator), create a temporary session/id
        is_orchestrator_call = session is None
        
        if is_orchestrator_call:
            # Create a temporary session just for this request
            from src.core.database import db_manager
            import uuid
            
            
            # Create a temporary session context
            async with db_manager.async_session_maker() as temp_session:
                response_text, _ = await self._process_internal(
                    temp_session, conversation_id, user_message, document_ids
                )
                
                # Format for TaskManager
                return {
                    "agent_message": response_text,
                    "agent_name": "RAGAgent"
                }
        else:
            # Internal call with existing session validation
            return await self._process_internal(
                session, conversation_id, user_message, document_ids
            )

    async def _process_internal(
        self,
        session: AsyncSession,
        conversation_id: str,
        user_message: str,
        document_ids: Optional[List[str]],
    ) -> tuple[str, int]:
        """Internal processing logic"""
        logger.info(f"[RAG Agent] Processing message for conversation {conversation_id}")
        logger.debug(f"[RAG Agent] User message: {user_message[:100]}...")
        logger.debug(f"[RAG Agent] Document IDs: {document_ids}")
        
        try:
            # Load conversation history
            # Note: For orchestrator/temp calls, this might be empty, which is fine
            logger.info(f"[RAG Agent] Loading conversation history for {conversation_id}")
            messages, total_tokens = await self.conversation_manager.prepare_context(
                session=session,
                conversation_id=conversation_id,
                new_message=user_message,
                retrieved_context=None,
            )
            
            history_count = len(messages) - 1  # Exclude the new message
            logger.info(f"[RAG Agent] Loaded {history_count} history messages, total context tokens: {total_tokens}")
            
            # Initialize state
            initial_state: RAGState = {
                "conversation_id": conversation_id,
                "user_message": user_message,
                "chat_history": messages[:-1],
                "retrieved_context": None,
                "generated_response": "",
                "document_ids": document_ids or [],
                "total_tokens": 0,
                "error": None,
            }
            
            logger.debug(f"[RAG Agent] Initialized state with {len(initial_state['chat_history'])} history messages")
            
            # Compile and run the graph
            compiled_graph = self.graph.compile()
            
            # Run the workflow
            logger.info(f"[RAG Agent] Starting workflow execution")
            final_state = await compiled_graph.ainvoke(initial_state)
            
            response = final_state["generated_response"]
            logger.info(f"[RAG Agent] Generated response ({len(response)} chars)")
            logger.debug(f"[RAG Agent] Response preview: {response[:200]}...")
            token_count = final_state["total_tokens"]
            logger.info(f"[RAG Agent] Total tokens used: {token_count}")
            
            if final_state.get("error"):
                logger.warning(f"[RAG Agent] Workflow completed with error: {final_state['error']}")
            
            return response, token_count
        
        except Exception as e:
            logger.error(f"[RAG Agent] Error processing message: {e}", exc_info=True)
            return "I apologize, but I encountered an error processing your message.", 0


# Global RAG agent instance
_rag_agent_instance: Optional[RAGAgent] = None


def get_rag_agent() -> RAGAgent:
    """Get or create the global RAG agent instance"""
    global _rag_agent_instance
    if _rag_agent_instance is None:
        _rag_agent_instance = RAGAgent()
    return _rag_agent_instance
