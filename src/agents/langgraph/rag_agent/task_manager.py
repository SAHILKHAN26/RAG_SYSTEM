from src.server.task_manager import InMemoryTaskManager
from src.agents.langgraph.rag_agent.agent import RAGAgent
from models.request import SendTaskRequest, SendTaskResponse
from models.task import Message, Task, TextPart, TaskStatus, TaskState
import json


class RAGAgentTaskManager(InMemoryTaskManager):
    """
    Connects the RAGAgent to the task handling system,
    managing inbound requests and outbound responses.
    """

    def __init__(self, agent: RAGAgent):
        super().__init__()
        self.agent = agent


    def _get_user_query(self, request: SendTaskRequest) -> str:
        """Extracts the user's text query from the incoming request."""
        # Simple extraction - assumes first part is text
        if request.params.message.parts and len(request.params.message.parts) > 0:
            return request.params.message.parts[0].text
        return ""

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """
        Processes an incoming task, invokes the RAG agent, and updates the task history.
        """
        print(f"Processing new RAG task: {request.params.id}")

        # Save or update the task in memory
        task = await self.upsert_task(request.params)

        # Extract the user's query
        query = self._get_user_query(request)
        
        # Invoke agent (Auto RAG mode via Orchestrator)
        # The agent.process_message now handles creating a temp session and returning a dict
        result = await self.agent.process_message(query)
        
        result_text = result['agent_message']
        agent_name = result['agent_name']

        # Format the agent's response as a Message object
        agent_message = Message(
            role="agent",
            parts=[TextPart(text=result_text)]
        )

        # Update the task status and history
        async with self.lock:
            task.status = TaskStatus(state=TaskState.COMPLETED)
            task.history.append(agent_message)
            task.agent_name = agent_name

        # Return the updated task
        return SendTaskResponse(id=request.id, result=task)
