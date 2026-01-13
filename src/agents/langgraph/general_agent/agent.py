"""
General Agent for open domain chat.

This module implements a simple agent for general conversation using LLM,
without document retrieval capabilities.
"""

from typing import TypedDict, List, Optional, Annotated, Any
from operator import add

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.services.llm_service import LLMService, LLMServiceFactory
from src.core.services.conversation_manager import ConversationManager, get_conversation_manager
from src.core.logger import app_logger as logger


# ============================================================================
# State Definition
# ============================================================================

class GeneralState(TypedDict):
    """State for the General agent workflow"""
    conversation_id: str
    user_message: str
    chat_history: Annotated[List[BaseMessage], add]
    generated_response: str
    total_tokens: int
    error: Optional[str]


# ============================================================================
# General Agent
# ============================================================================

class GeneralAgent:
    """Simple agent for open domain conversation"""
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        conversation_manager: Optional[ConversationManager] = None,
    ):
        """
        Initialize the General agent.
        
        Args:
            llm_service: LLM service for generation
            conversation_manager: Conversation manager for history
        """
        self.llm_service = llm_service or LLMServiceFactory.get_service()
        self.conversation_manager = conversation_manager or get_conversation_manager(self.llm_service)
        
        # Build the LangGraph workflow
        self.graph = self._build_graph()
        
        logger.info("General agent initialized")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        # Create the graph
        workflow = StateGraph(GeneralState)
        
        # Add nodes
        workflow.add_node("generate", self._generation_node)
        
        # Add edges
        workflow.set_entry_point("generate")
        workflow.add_edge("generate", END)
        
        return workflow
    
    async def _generation_node(self, state: GeneralState) -> GeneralState:
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
            
            # System prompt for general chat
            system_prompt = "You are a helpful AI assistant. Answer the user's questions to the best of your ability. Be helpful, harmless, and honest."
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
        user_message: str,
        session: Optional[AsyncSession] = None,
        conversation_id: Optional[str] = None,
    ) -> Any:
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
            
            # Use ephemeral conversation ID
            conversation_id = f"temp-general-{uuid.uuid4()}"
            
            # Create a temporary session context
            async with db_manager.async_session_maker() as temp_session:
                response_text, _ = await self._process_internal(
                    temp_session, conversation_id, user_message
                )
                
                # Format for TaskManager
                return {
                    "agent_message": response_text,
                    "agent_name": "GeneralAgent"
                }
        else:
            # Internal call with existing session validation
            return await self._process_internal(
                session, conversation_id, user_message
            )

    async def _process_internal(
        self,
        session: AsyncSession,
        conversation_id: str,
        user_message: str,
    ) -> tuple[str, int]:
        """Internal processing logic"""
        logger.info(f"Processing message for conversation {conversation_id}")
        
        try:
            # Load conversation history
            messages, _ = await self.conversation_manager.prepare_context(
                session=session,
                conversation_id=conversation_id,
                new_message=user_message,
                retrieved_context=None,
            )
            
            # Initialize state
            initial_state: GeneralState = {
                "conversation_id": conversation_id,
                "user_message": user_message,
                "chat_history": messages[:-1],
                "generated_response": "",
                "total_tokens": 0,
                "error": None,
            }
            
            # Compile and run the graph
            compiled_graph = self.graph.compile()
            
            # Run the workflow
            final_state = await compiled_graph.ainvoke(initial_state)
            
            response = final_state["generated_response"]
            logger.info(f"Final response from GENERAL Agent: {response}")
            token_count = final_state["total_tokens"]
            
            if final_state.get("error"):
                logger.warning(f"Workflow completed with error: {final_state['error']}")
            
            return response, token_count
        
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return "I apologize, but I encountered an error processing your message.", 0


# Global General agent instance
_general_agent_instance: Optional[GeneralAgent] = None


def get_general_agent() -> GeneralAgent:
    """Get or create the global General agent instance"""
    global _general_agent_instance
    if _general_agent_instance is None:
        _general_agent_instance = GeneralAgent()
    return _general_agent_instance
