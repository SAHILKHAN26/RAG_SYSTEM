import os
from dotenv import load_dotenv
from src.server.server import A2AServer
from models.agent import AgentCard, AgentCapabilities, AgentSkill
from src.agents.langgraph.rag_agent.task_manager import RAGAgentTaskManager
from src.agents.langgraph.rag_agent.agent import RAGAgent

def main():
    """
    Starts the RAG LangGraph agent server, making it accessible via HTTP.
    """
    load_dotenv()

    host = os.getenv("HOST", "localhost")
    
    # Use port 10002 for RAG Agent (different from NLP to Cypher)
    try:
        port = int(os.getenv("RAG_AGENT_PORT", 10002))
    except ValueError:
        print("Warning: PORT environment variable is not a valid integer. Using default 10002.")
        port = 10002

    # Define capabilities
    capabilities = AgentCapabilities(streaming=False)

    # Define the skill this RAG agent offers
    skill = AgentSkill(
        id="RAG_agent_skill",
        name="RAG Knowledge Base Skill",
        description=(
            "Retrieves identifying information from uploaded documents and knowledge base. "
            "Capable of searching across multiple file formats (PDF, DOCX, TXT) and synthesizing answers."
        ),
        tags=["RAG", "knowledge base", "documents", "search", "retrieval", "context"],
        examples=[
            "What does the document say about project timelines?",
            "Summarize the key points from the uploaded report",
            "Search the knowledge base for 'error handling'",
            "Explain the architecture described in the files"
        ]
    )

    # Create an agent card
    agent_card = AgentCard(
        name="RAGAgent",
        description=(
            "A Retrieval-Augmented Generation agent that combines vector search "
            "with LLM capabilities to answer questions based on your documents."
        ),
        url=f"http://{host}:{port}/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=capabilities,
        skills=[skill]
    )

    # Initialize and start the A2A server
    server = A2AServer(
        host=host,
        port=port,
        agent_card=agent_card,
        task_manager=RAGAgentTaskManager(agent=RAGAgent())
    )

    server.start()

if __name__ == "__main__":
    main()
