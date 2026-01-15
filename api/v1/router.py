from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
import httpx
import uuid
import asyncio
import json
from datetime import datetime
from typing import Dict, Optional, List
from fastapi.responses import PlainTextResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.logger import app_logger as logger
from src.core.constant import ORCHESTRATOR_HOST,ORCHESTRATOR_PORT
from src.core.database import get_db, Conversation, Message, MessageRole, User
from models.request import DirectCallRequest, OrchestratorRequest

orchestrator_router = APIRouter()
agent_list_router=  APIRouter()
direct_agent_router = APIRouter()




def build_orchestrator_payload(
    query: str,
    target_agents: Optional[List[str]] = None,
    toggle_all: bool = False,
    conversation_id: str = None
) -> Dict:

    payload = {
        "jsonrpc": "2.0",
        "method": "tasks/send",
        "params": {
            "id": str(uuid.uuid4()),
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": query}]
            },
            "metadata": {"toggle_all": toggle_all, "conversation_id":conversation_id  }
        },
        "id": 1
    }

    if target_agents:
        payload["params"]["metadata"]["target_agents"] = target_agents
        logger.debug(f"Routing request to specific agents: {target_agents}")

    logger.info(f"Built orchestrator payload: {payload}")
    return payload


@orchestrator_router.post("/global-chat", summary="Query the agent orchestrator")
async def call_orchestrator(
    request: OrchestratorRequest,
    session: AsyncSession = Depends(get_db)  # Inject DB session
):
    logger.info(f"Starting the agent call as query received as {request.query}")
    if not request.query.strip():
        logger.warning("Received empty query")
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # --- Persistence Logic Start ---
        user_id = request.user_id
        conversation_id = request.conversation_id
        
        # Verify user exists
        user_result = await session.execute(
            select(User).where(User.id == user_id)
        )
        if not user_result.scalar_one_or_none():
            logger.warning(f"User {user_id} not found")
            raise HTTPException(status_code=404, detail="User not found")

        # Ensure we have a conversation ID if not provided
        if not conversation_id:
             conversation_id = str(uuid.uuid4())
             # Update request object to use this ID for the orchestrator call
             request.conversation_id = conversation_id

        # Check if conversation exists
        existing_conv_result = await session.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        existing_conversation = existing_conv_result.scalar_one_or_none()
        
        next_seq = 0
        if existing_conversation:
             # Get next sequence number
            seq_result = await session.execute(
                select(func.max(Message.sequence_number)).where(
                    Message.conversation_id == conversation_id
                )
            )
            max_seq = seq_result.scalar() or -1
            next_seq = max_seq + 1
        else:
            # Create new conversation
            logger.info(f"Creating new conversation: {conversation_id}")
            
            # Simplified: Create conversation
            conversation = Conversation(
                id=conversation_id,
                user_id=user_id, # Potential FK issue if this user doesn't exist
                title="Global Chat Conversation"
            )
            session.add(conversation)
            await session.flush()
            existing_conversation = conversation # Treat as existing for subsequent updates

        # Add User Message
        user_msg = Message(
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=request.query,
            sequence_number=next_seq
        )
        session.add(user_msg)
        await session.flush()
        # --- Persistence Logic End (Pre-Call) ---

        url = f"http://{ORCHESTRATOR_HOST}:{ORCHESTRATOR_PORT}/"
        payload = build_orchestrator_payload(
            request.query,
            request.target_agents,
            request.toggle_all,
            conversation_id # Pass the resolved ID
        )

        logger.info("Sending request to orchestrator")
        logger.debug(f"Orchestrator URL: {url}")

        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=180.0)

        resp.raise_for_status()
        full_agent_response = resp.json()

        logger.info("Received response from orchestrator")
        logger.debug(f"Full orchestrator response: {full_agent_response}")

    

        # --- Persistence Logic Start (Post-Call) ---
        # Add Assistant Message
        # Estimate tokens (simple estimation or get from service)
        response_content = json.dumps(full_agent_response)
        token_count = len(response_content) // 4 # Rough approx
        

        result = full_agent_response.get("result")
        if not result:
            logger.error("Missing 'result' in orchestrator response")
            raise HTTPException(500, "Malformed orchestrator response")

        agent_name = result.get("agent_name")
        history = result.get("history", [])

        agent_text = (
            history[-1]["parts"][0].get("text")
            if history and history[-1].get("parts")
            else None
        )

        if not agent_text:
            logger.warning("Agent returned empty text response")
            return PlainTextResponse(
                content="No meaningful response received from agent."
            )

        if agent_name == "Multi-Agent Fan-Out":
            try:
                logger.info("Processing multi-agent fan-out response")
                # For fan-out, we store the raw JSON string of the agents' responses
                # agent_text is already a JSON string provided by the orchestrator
                assistant_msg = Message(
                    conversation_id=conversation_id,
                    role=MessageRole.ASSISTANT,
                    content=agent_text, 
                    token_count=token_count,
                    sequence_number=next_seq + 1
                )
                session.add(assistant_msg)
            
            except Exception as e:
                logger.error("Error processing multi-agent response", exc_info=True)
                raise HTTPException(500, "Invalid multi-agent response processing")
        else:
            # Single agent case
            assistant_msg = Message(
                conversation_id=conversation_id,
                role=MessageRole.ASSISTANT,
                content=agent_text, 
                token_count=token_count,
                sequence_number=next_seq + 1
            )
            session.add(assistant_msg)

        # Update conversation stats
        if existing_conversation:
             existing_conversation.total_tokens += token_count
             existing_conversation.updated_at = datetime.utcnow()
        
        await session.commit()
        logger.info("Persisted messages and updated conversation")

        if agent_name == "Multi-Agent Fan-Out":
             return json.loads(agent_text)
        
        return {"agent_name" : agent_name, "agent_message": agent_text, "conversation_id": conversation_id}  


    except httpx.HTTPStatusError as exc:
        logger.error(
            f"Orchestrator HTTP error: {exc.response.status_code}",
            exc_info=True
        )
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=exc.response.text
        )

    except httpx.RequestError as exc:
        logger.error("Failed to reach orchestrator", exc_info=True)
        raise HTTPException(
            status_code=502,
            detail="Unable to contact orchestrator service"
        )
    except Exception as e:
        logger.error(f"Error in global chat: {e}", exc_info=True)
        await session.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")

@agent_list_router.get("/agents", summary="List all agents")
async def list_available_agents():

    url = f"http://{ORCHESTRATOR_HOST}:{ORCHESTRATOR_PORT}/agents"
    logger.info("Fetching agent list from orchestrator")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10.0)

        resp.raise_for_status()
        data = resp.json()

        agents = data.get("agents", [])
        if not isinstance(agents, list):
            logger.error(f"Malformed agent list response: {data}")
            return []

        logger.info(f"Found {len(agents)} agents")
        return agents

    except httpx.RequestError:
        logger.error("Unable to reach orchestrator for agent list", exc_info=True)
        raise HTTPException(502, "Unable to fetch agent list")

    except httpx.HTTPStatusError as exc:
        logger.error("Agent list API failed", exc_info=True)
        raise HTTPException(exc.response.status_code, exc.response.text)

