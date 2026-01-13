from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import httpx
import uuid
import asyncio
import json
from typing import Dict, Optional, List
from fastapi.responses import PlainTextResponse
from src.core.logger import app_logger as logger
from src.core.constant import ORCHESTRATOR_HOST,ORCHESTRATOR_PORT
from models.request import DirectCallRequest, OrchestratorRequest

orchestrator_router = APIRouter()
agent_list_router=  APIRouter()
direct_agent_router = APIRouter()




def build_orchestrator_payload(
    query: str,
    target_agents: Optional[List[str]] = None,
    toggle_all: bool = False
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
            "metadata": {"toggle_all": toggle_all}
        },
        "id": 1
    }

    if target_agents:
        payload["params"]["metadata"]["target_agents"] = target_agents
        logger.debug(f"Routing request to specific agents: {target_agents}")

    logger.info(f"Built orchestrator payload: {payload}")
    return payload


@orchestrator_router.post("/data-privacy/call", summary="Query the agent orchestrator")
async def call_orchestrator(request: OrchestratorRequest):
    logger.info(f"Starting the agent call as query received as {request.query}")
    if not request.query.strip():
        logger.warning("Received empty query")
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    url = f"http://{ORCHESTRATOR_HOST}:{ORCHESTRATOR_PORT}/"
    payload = build_orchestrator_payload(
        request.query,
        request.target_agents,
        request.toggle_all
    )

    logger.info("Sending request to orchestrator")
    logger.debug(f"Orchestrator URL: {url}")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=180.0)

        resp.raise_for_status()
        full_agent_response = resp.json()

        logger.info("Received response from orchestrator")
        logger.debug(f"Full orchestrator response: {full_agent_response}")

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
                return json.loads(agent_text)
            except json.JSONDecodeError:
                logger.error("Invalid JSON from multi-agent response", exc_info=True)
                raise HTTPException(500, "Invalid multi-agent response format")

        # return PlainTextResponse(content=agent_text)
        return {"agent_name" : agent_name, "agent_message": agent_text}  

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

@direct_agent_router.post("/agent_call", summary= "Call one or more agents directly in parallel")
async def direct_agent_call(request: Request,  body: DirectCallRequest):
    """Bypasses the orchestrator and sends a query directly to one or more 
    specified agents in parallel. Returns an aggregated result."""

    agent_registry  = request.app.state.agent_registry
    tasks = []
    async with httpx.AsyncClient() as client:
        for agent_name in body.target_agents:
            agent_url = agent_registry.get(agent_name)
            if not agent_url:
                logger.warning(f"Agent '{agent_name}' not found in registry, skipping")
                continue

            payload = build_orchestrator_payload(body.query)
            task = client.post(agent_url, json=payload, timeout=180.0)
            tasks.append(task)

        if not tasks:
            raise HTTPException(
                status_code= 404,
                detail= f"None of the specified agents were found. Available agents are: {list(agent_registry.keys())}"
            )
        
        responses = await asyncio.gather(*tasks, return_exceptions= True)
    
    aggregated_results = {}
    for agent_name, response in zip(body.target_agents, responses):
        if isinstance(response, Exception):
            aggregated_results[agent_name] = {"error": f"Failed to get response: {str(response)}"}
        else:
            try: 
                response.raise_for_status()
                full_agent_response = response.json()
                agent_text = full_agent_response['result']['history'][-1]['parts'][0]['text']
                aggregated_results[agent_name] = {"answer": agent_text or "Agent returned an empty response."}

            except Exception as e:
                # If parsing fails or status is bad, record that error
                aggregated_results[agent_name] = {"error": f"Could not parse response: {str(e)}"}

    # Return the final dictionary containing all results
    return aggregated_results
