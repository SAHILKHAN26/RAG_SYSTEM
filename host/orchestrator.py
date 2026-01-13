import os                           # Standard library for interacting with the operating system
import uuid                         # For generating unique identifiers (e.g., session IDs)
import asyncio
# -----------------------------------------------------------------------------
# Google ADK / Gemini imports
# -----------------------------------------------------------------------------
from google.adk.agents.llm_agent import LlmAgent
# LlmAgent: core class to define a Gemini-powered AI agent
from google.adk.sessions import InMemorySessionService
# InMemorySessionService: stores session state in memory (for simple demos)
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
# InMemoryMemoryService: optional conversation memory stored in RAM
from google.adk.artifacts import InMemoryArtifactService
# InMemoryArtifactService: handles file/blob artifacts (unused here)
from google.adk.runners import Runner
# Runner: orchestrates agent, sessions, memory, and tool invocation
from google.adk.agents.readonly_context import ReadonlyContext
# ReadonlyContext: passed to system prompt function to read context
from google.adk.tools.tool_context import ToolContext
# ToolContext: passed to tool functions for state and actions
from google.genai import types           
# types.Content & types.Part: used to wrap user messages for the LLM
# -----------------------------------------------------------------------------
# A2A server-side infrastructure
# -----------------------------------------------------------------------------
from src.server.task_manager import InMemoryTaskManager
# InMemoryTaskManager: base class providing in-memory task storage and locking
from models.request import SendTaskRequest, SendTaskResponse
# Data models for incoming task requests and outgoing responses
from models.task import Message, TaskStatus, TaskState, TextPart
# Message: encapsulates role+parts; TaskStatus/State: status enums; TextPart: text payload
# -----------------------------------------------------------------------------
# Connector to child A2A agents
# -----------------------------------------------------------------------------
from host.agent_connect import AgentConnector
# AgentConnector: lightweight wrapper around A2AClient to call other agents
from models.agent import AgentCard
# AgentCard: metadata structure for agent discovery results
from dotenv import load_dotenv
import json
# Load environment variables (e.g., API keys) from a .env file
load_dotenv()


class OrchestratorAgent:
    """
    🤖 Uses a Gemini LLM to route incoming user queries,
    calling out to any discovered child A2A agents via tools.
    """

    # Define supported MIME types for input/output
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self, agent_cards: list[AgentCard]):
        # Build one AgentConnector per discovered AgentCard
        # agent_cards is a list of AgentCard objects returned by discovery
        self.connectors = {
            card.name: AgentConnector(card.name, card.url)
            for card in agent_cards
        }

        # Build the internal LLM agent with our custom tools and instructions
        self._agent = self._build_agent()

        # Static user ID for session tracking across calls
        self._user_id = "orchestrator_user"

        # Runner wires up sessions, memory, artifacts, and handles agent.run()
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    def _get_agent_name_from_events(self,events):
        for event in events:
            if not event.content or not event.content.parts:
                continue
            for part in event.content.parts:
                if part.function_call and part.function_call.args:
                    # If args is already a dict, access directly
                    args = part.function_call.args
                    if isinstance(args, dict):
                        return args.get("agent_name", "")
                    # If args is a JSON string, parse it
                    try:
                        args_dict = json.loads(args)
                        return args_dict.get("agent_name", "")
                    except Exception:
                        continue
        return ""

    def _build_agent(self) -> LlmAgent:
        """
        Construct the Gemini-based LlmAgent with:
        - Model name
        - Agent name/description
        - System instruction callback
        - Available tool functions
        """
        return LlmAgent(
            model="gemini-2.5-pro",    # Specify Gemini model version
            name="orchestrator_agent",          # Human identifier for this agent
            description="Delegates user queries to child A2A agents based on intent.",
            instruction=self._root_instruction,  # Function providing system prompt text
            tools=[
                self._list_agents,               # Tool 1: list available child agents
                self._delegate_task             # Tool 2: call a child agent
            ],
        )

    def _root_instruction(self, context: ReadonlyContext) -> str:
        """
        System prompt function: returns instruction text for the LLM,
        including which tools it can use and a list of child agents.
        """
        # Build a bullet-list of agent names
        agent_list = "\n".join(f"- {name}" for name in self.connectors)
        return (
           "You are an orchestrator with multiple agents or orchestrator:\n"
            "1) list_agents() -> list available child agents\n"
            "2) delegate_task(agent_name, message) -> call that agent\n\n"
            
            f"Available agents:\n{agent_list}\n\n"
            "Use DPOrchestratorAgent when the passed query seems like a article or news and it contains the incident information related to any data breach related to any company"
            "Use GraphRagOrchestratorAgent when the query is related to asking any incident details or and company related details"
            "Use SuggestionAgent when the user wants to get some suggestion about any company or any incident"        
            "If you decide some agent and the agent is not in the agent list or got some error then just return some apology message"  )
        

    def _list_agents(self) -> list[str]:
        """
        Tool function: returns the list of child-agent names currently registered.
        Called by the LLM when it wants to discover available agents.
        """
        return list(self.connectors.keys())


    async def delegate_task_directly(self, agent_name:  str, user_text: str, session_id: str)-> dict:
        """Directly delegates a task to a specific agent without LLM routing."""
        
        print(f"Directly delegating to {agent_name}")
        if agent_name not in self.connectors:
            return {
                'agent_name': 'OrchestratorAgent',
                'agent_message': f"Error: The specified agent '{agent_name} is not available."
            }
        connector = self.connectors[agent_name]

        # delegate task and get the result
        child_task = await connector.send_task(user_text, session_id)
        response_text = ""
        if child_task.history and len(child_task.history) >0:
            last_part = child_task.history[-1].parts[0]
            if last_part and hasattr(last_part, 'text'):
                response_text = last_part.text

        return {
            'agent_name': agent_name,
            'agent_message': response_text
        }


    async def _delegate_task(
        self,
        agent_name: str,
        message: str,
        tool_context: ToolContext
    ) -> str:
        """
        Tool function: forwards the `message` to the specified child agent
        (via its AgentConnector), waits for the response, and returns the
        text of the last reply.
        """
        # Validate agent_name exists
        if agent_name not in self.connectors:
            return "The required agent is not present to call, hence returning"
        connector = self.connectors[agent_name]

        # Ensure session_id persists across tool calls via tool_context.state
        state = tool_context.state
        if "session_id" not in state:
            state["session_id"] = str(uuid.uuid4())
        session_id = state["session_id"]

        # Delegate task asynchronously and await Task result
        child_task = await connector.send_task(message, session_id)

        # Extract text from the last history entry if available
        if child_task.history and len(child_task.history) > 1:
            return child_task.history[-1].parts[0].text
        return ""

    async def invoke(self, query: str, session_id: str) -> dict:
        """
        Main entry: receives a user query + session_id,
        sets up or retrieves a session, wraps the query for the LLM,
        runs the Runner (with tools enabled), and returns the final text.
        """
        # Attempt to reuse an existing session
        session = await self._runner.session_service.get_session(
            app_name=self._agent.name,
            user_id=self._user_id,
            session_id=session_id
        )

        # Create new if not found
        if session is None:
            session = await self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                session_id=session_id,
                state={}
            )
        # Wrap the user query in a types.Content message
        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=query)]
        )
        # Run the agent synchronously; collects a list of events
        events = list(self._runner.run(
            user_id=self._user_id,
            session_id=session.id,
            new_message=content
        ))
        # If no content or parts, return empty fallback
        if not events or not events[-1].content or not events[-1].content.parts:
            return ""
        agent_name = self._get_agent_name_from_events(events)
        # Join all text parts into a single string reply
        agent_reply = "\n".join(
            p.text for p in events[-1].content.parts if p.text)
        return {'agent_name':agent_name,'agent_message':agent_reply}


class OrchestratorTaskManager(InMemoryTaskManager):
    """
    🪄 TaskManager wrapper: exposes OrchestratorAgent.invoke() over the
    A2A JSON-RPC `tasks/send` endpoint, handling in-memory storage and
    response formatting.
    """
    def __init__(self, agent: OrchestratorAgent):
        super().__init__()       # Initialize base in-memory storage
        self.agent = agent       # Store our orchestrator logic

    def _get_user_text(self, request: SendTaskRequest) -> str:
        """
        Helper: extract the user's raw input text from the request object.
        """
        return request.params.message.parts[0].text

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """
        Called by the A2A server when a new task arrives:
        1. Store the incoming user message
        2. Invoke the OrchestratorAgent to get a response
        3. Append response to history, mark completed
        4. Return a SendTaskResponse with the full Task
        """
        print(f"OrchestratorTaskManager received task {request.params.id}")

        # Step 1: save the initial message
        task = await self.upsert_task(request.params)

        # Step 2: run orchestration logic
        user_text = self._get_user_text(request)
        session_id  = request.params.sessionId

        # the three-tired routing logic
        target_agents = []
        toggle_all = False
        if request.params.metadata:
            target_agents = request.params.metadata.get("target_agents", [])
            toggle_all = request.params.metadata.get("toggle_all", False)

        response_data = None 
        # TIER 1: SINGLE AGENT SELECTED
        if len(target_agents)==1:
            selected_agent_name = target_agents[0]
            print(f"[Orchestrator Logic] Tier 1: Answer from the selected agent: '{selected_agent_name}'")

            response_data = await self.agent.delegate_task_directly(
                agent_name=selected_agent_name,
                user_text=user_text,
                session_id=session_id
            )

        # TIER 2: MULTIPLE AGENTS SELECTED
        elif len(target_agents)>1:

            # CASE A: toggle_all is TRUE -> Get the SINGLE BEST answer
            if toggle_all is True:
                print(f"[Orchestrator Logic] Tier 2: Best Answer from multiple selected agents: {target_agents}")

                agent_list_str = ", ".join(target_agents)
                instruction = (f"You must choose the single best agent from the following list to answer the user's query: [{agent_list_str}]"
                               "Do not choose any other agent."
                               )
                query_for_llm = instruction + f"User query:- '{user_text}"
                response_data = await self.agent.invoke(query_for_llm, session_id)

            # CASE B: toggle_all is FALSE -> Get answers from ALL agents
            else:
                print(f"[Orchestrator Logic] Tier 2: Answers from all selected agents: {target_agents}")
                tasks = []
                for agent_name in target_agents:
                    task_coro = self.agent.delegate_task_directly(
                        agent_name=agent_name,
                        user_text=user_text,
                        session_id=session_id
                    )
                    tasks.append(task_coro)
                
                results = await asyncio.gather(*tasks)
                
                aggregated_message = {}
                for res in results:
                    agent_name = res.get('agent_name', 'Unknown Agent')
                    agent_message = res.get('agent_message', 'No response.')
                    aggregated_message[agent_name]= agent_message

                json_string_message = json.dumps(aggregated_message, indent=2)
                response_data = {
                    'agent_name': 'Multi-Agent Fan-Out',
                    'agent_message': json_string_message
                }

        # TIER 3: NO AGENT IS SELECTED
        else:
            print("[Orchestrator Logic] Tier 3: Automatic Routing from all agents.")
            response_data = await self.agent.invoke(user_text, session_id)


        # Step 3: wrap the LLM output into a Message
        reply = Message(role="agent", parts=[TextPart(
            text=response_data['agent_message'])])
        
        print("Agent response in O-TASK_MANAGER:",reply)
        async with self.lock:
                task.status = TaskStatus(state=TaskState.COMPLETED)
                task.agent_name = response_data['agent_name']
                task.history.append(reply)

        
        # Step 4: return structured response
        return SendTaskResponse(id=request.id, result=task)
