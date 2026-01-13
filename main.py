from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from api.v1.router import orchestrator_router, agent_list_router, direct_agent_router
from api.v1.conversation_routes import conversation_router, user_router, document_router


from src.core.constant import SERVER_HOST,SERVER_PORT



app = FastAPI(title = "Agent gateway API", version="1.0.0")

@app.get("/")
async def index(request: Request):
    return {"status":"running"}



# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Include API router
app.include_router(orchestrator_router, prefix="/orchestrator", tags=["orchestrator"])
app.include_router(agent_list_router, prefix="/a2a", tags =["agent_list"])
app.include_router(direct_agent_router, prefix="/direct", tags=["Direct Agent Call"])

# Include new conversation API routers
app.include_router(conversation_router, prefix="/api/v1", tags=["Conversations"])
app.include_router(user_router, prefix="/api/v1", tags=["Users"])
app.include_router(document_router, prefix="/api/v1", tags=["Documents"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_HOST, port=int(SERVER_PORT))