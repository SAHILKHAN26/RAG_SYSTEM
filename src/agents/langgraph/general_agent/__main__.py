import os
from dotenv import load_dotenv
from src.server.server import A2AServer
from models.agent import AgentCard, AgentCapabilities, AgentSkill
from src.agents.langgraph.general_agent.task_manager import GeneralAgentTaskManager
from src.agents.langgraph.general_agent.agent import GeneralAgent

def main():
    """
    Starts the General LangGraph agent server, making it accessible via HTTP.
    """
    load_dotenv()

    host = os.getenv("HOST", "localhost")
    
    # Use port 10003 for General Agent
    try:
        port = int(os.getenv("GENERAL_AGENT_PORT", 10003))
    except ValueError:
        print("Warning: PORT environment variable is not a valid integer. Using default 10003.")
        port = 10003

    # Define capabilities
    capabilities = AgentCapabilities(streaming=False)

    # Define the skill this General agent offers
    skill = AgentSkill(
        id="General_agent_skill",
        name="Open Domain Chat Skill",
        description=(
            "Provides general conversational assistance, answering questions, giving advice, "
            "and helping with tasks that don't require specific document retrieval."
        ),
        tags=["general", "chat", "assistant", "help"],
        examples=[
            "Hi, how are you?",
            "Write a poem about coding",
            "Explain quantum computing simply",
            "Give me some ideas for a blog post"
        ]
    )

    # Create an agent card
    agent_card = AgentCard(
        name="GeneralAgent",
        description=(
            "A general-purpose AI assistant for open conversations and general knowledge."
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
        task_manager=GeneralAgentTaskManager(agent=GeneralAgent())
    )

    server.start()

if __name__ == "__main__":
    main()
