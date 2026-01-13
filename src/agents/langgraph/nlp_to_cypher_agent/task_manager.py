from src.server.task_manager import InMemoryTaskManager
from src.agents.langgraph.nlp_to_cypher_agent.agent import RAGAgent
from models.request import SendTaskRequest, SendTaskResponse
from models.task import Message, Task, TextPart, TaskStatus, TaskState
import json


class RAGAgentTaskManager(InMemoryTaskManager):
    """
    Connects the NLPCypherLangGraphAgent to the task handling system,
    managing inbound requests and outbound responses.
    """

    def __init__(self, agent: RAGAgent):
        super().__init__()
        self.agent = agent


    def _get_user_query(self, request: SendTaskRequest) -> str:
        """Extracts the user's text query from the incoming request."""
        return request.params.message.parts[0].text

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """
        Processes an incoming task, invokes the NLPCypher agent, and updates the task history.
        """
        print(f"Processing new NLPCypher task: {request.params.id}")

        # Save or update the task in memory
        task = await self.upsert_task(request.params)

        # Extract the user's query
        query = self._get_user_query(request)
        # agent_name = self.agent.agent_name
        result_text = await self.agent.process_message(query)

        # Format the agent's response as a Message object
        agent_message = Message(
            role="agent",
            parts=[TextPart(text=result_text['agent_message'])]
        )

        # Update the task status and history
        async with self.lock:
            task.status = TaskStatus(state=TaskState.COMPLETED)
            task.history.append(agent_message)
            task.agent_name = result_text['agent_name']

        # Return the updated task
        return SendTaskResponse(id=request.id, result=task)
