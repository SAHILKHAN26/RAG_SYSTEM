import asyncio                              # Built-in for running async coroutines
import click                                # Library for building CLI interfaces
import sys
import os

# Add project root to sys.path to allow imports from src and models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Utility for discovering remote A2A agents from a local registry
from src.utils.discovery import DiscoveryClient
# Shared A2A server implementation (Starlette + JSON-RPC)
from src.server.server import A2AServer
# Pydantic models for defining agent metadata (AgentCard, etc.)
from models.agent import AgentCard, AgentCapabilities, AgentSkill
# Orchestrator implementation and its task manager
from host.orchestrator import (
    OrchestratorAgent,
    OrchestratorTaskManager
)
from src.core.constant import ORCHESTRATOR_HOST, ORCHESTRATOR_PORT, HOST_REGISTRY_FILE



def main(host: str = ORCHESTRATOR_HOST, port: int = ORCHESTRATOR_PORT, registry: str = HOST_REGISTRY_FILE ):
    """
    Entry point to start the OrchestratorAgent A2A server.

    Steps performed:
    1. Load child-agent URLs from the registry JSON file.
    2. Fetch each agent's metadata via `/.well-known/agent.json`.
    3. Instantiate an OrchestratorAgent with discovered AgentCards.
    4. Wrap it in an OrchestratorTaskManager for JSON-RPC handling.
    5. Launch the A2AServer to listen for incoming tasks.
    """
    # 1) Discover all registered child agents from the registry file
    discovery = DiscoveryClient(registry_file=registry)
    # Run the async discovery synchronously at startup
    agent_cards = asyncio.run(discovery.list_agent_cards())
   
    # Warn if no agents are found in the registry
    if not agent_cards:
        print(
            "No agents found in registry – the orchestrator will have nothing to call"
        )

    # 2) Define the OrchestratorAgent's own metadata for discovery
    capabilities = AgentCapabilities(streaming=False)
    skill = AgentSkill(
        id="orchestrate",                          # Unique skill identifier
        name="Orchestrate Tasks",                  # Human-friendly name
        description=(
                "This skill routes user queries to the appropriate specialized agent based on intent:\n\n"
                 "Route the query to available agents"
        ),
        tags=["routing", "orchestration"],       # Keywords to aid discovery
        examples=[                                  # Sample user queries
            "tell me about the company x?",
            "tell me about the company who recently gone through data breach?"
            "what is sql injection ?"
            "which company has most breaches in last 6 months?"
        ]
    )
    orchestrator_card = AgentCard(
        name="OrchestratorAgent",
        description=(
            "This skill routes user queries to the appropriate specialized agent based on intent:\n\n"
            "Route the query to available agents"
            ),
        url=f"http://{host}:{port}/",             # Public endpoint
        version="1.0.0",
        defaultInputModes=["text"],                # Supported input modes
        defaultOutputModes=["text"],               # Supported output modes
        capabilities=capabilities,
        skills=[skill]
    )

    # 3) Instantiate the OrchestratorAgent and its TaskManager
    orchestrator = OrchestratorAgent(agent_cards=agent_cards)
    task_manager = OrchestratorTaskManager(agent=orchestrator)

    # 4) Create and start the A2A server
    server = A2AServer(
        host=host,
        port=port,
        agent_card=orchestrator_card,
        task_manager=task_manager,
        discovered_agent_cards = agent_cards
    )
    server.start()


if __name__ == "__main__":
    main()