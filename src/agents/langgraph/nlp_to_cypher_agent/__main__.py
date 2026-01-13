import os
from dotenv import load_dotenv
from src.server.server import A2AServer
from models.agent import AgentCard, AgentCapabilities, AgentSkill
from src.agents.langgraph.nlp_to_cypher_agent.task_manager import RAGAgentTaskManager
from src.agents.langgraph.nlp_to_cypher_agent.agent import RAGAgent

# @click.command()
# @click.option("--host", default="localhost", help="Host to bind the server to")
# @click.option("--port", default=18092, help="Port number for the server")
def main():
    """
    Starts the NLPCypher LangGraph agent server, making it accessible via HTTP.
    """
    load_dotenv()

    
    host = os.getenv("HOST", "localhost")
    
    # Fetch PORT, default to 18096 if not set. Convert to integer.
    try:
        port = int(os.getenv("NLPCypher_AGENT_PORT", 10001))
    except ValueError:
        print("Warning: PORT environment variable is not a valid integer. Using default 10001.")
        port = 10001

    # Define capabilities
    capabilities = AgentCapabilities(streaming=False)

    # Define the skill this NLPCypher agent offers
    skill = AgentSkill(
        id="NLPCypher_agent_skill",
        name="NLPCypher Planning Agent Skill",
        description=(
            "Performs NLP To Cypher to get data related to company, data breaches, incident, risk factors "
            "This agent uses LangGraph to interact with tools like NLPCyphers, "
        ),
        tags=["NLPCypher", "company", "incident", "data breaches", "risk factors", "services", "control", "third party"],
        examples=[
            "Give me total data breaches list",
            "give me incident happened with amazon company",
            "What is the risk factor of Uber Technologies and impacted third party",
            "Name the services given by Yahoo third party",
            "give me history of risk scores of Salesforce third party"
        ]
    )

    # Create an agent card
    agent_card = AgentCard(
        name="RAGAgent",
        description=(
            "This agent helps to get information related to company or third party or incident or data breaches or services"
            "by using nlp to cypher process ."
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
