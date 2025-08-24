# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from uuid import uuid4
import time # <-- Added import for timestamp
from pydantic import BaseModel # <-- Added import for response model

# Import the LangGraph components from our new graph.py file
from Graph.pipeline import rag_graph, retriever, run_query
from langchain.memory import ConversationBufferMemory
# This import seems unused in this file, but keeping it as requested
from LLM.llm import LLMClient

# ---------- FastAPI App ----------
app = FastAPI(title="LangGraph RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dictionary to manage memory for multiple users
user_memories: Dict[str, ConversationBufferMemory] = {}


# ---------- ADDED SECTION: The missing /status endpoint ----------
# This response model ensures our status endpoint has a consistent structure
class StatusResponse(BaseModel):
    status: str
    timestamp: float

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """A simple endpoint for the frontend to check if the server is running."""
    return {"status": "ok", "timestamp": time.time()}
# ----------------------------------------------------------------


# POST /start (starts a new conversation)
@app.post("/start")
async def start_conversation(request: Dict[str, str]):
    """Starts a new conversation session, generating a new user ID."""
    global user_memories
    
    user_query = request.get("user_query")
    if not user_query:
        raise HTTPException(status_code=400, detail="'user_query' is required.")

    user_id = str(uuid4())
    user_memories[user_id] = ConversationBufferMemory(return_messages=True)
    
    # Call the run_query function from the graph.py module
    output = run_query(rag_graph, retriever, user_id, user_query, user_memories[user_id])
    assistant_response = output.get("final_answer", "Sorry, something went wrong.")
    
    return {
        "message": assistant_response,
        "user_id": user_id
    }

# POST /continue (adds a follow-up turn)
@app.post("/continue")
async def continue_conversation(request: Dict[str, str]):
    """Continues an existing conversation session using the provided user ID."""
    global user_memories
    
    user_id = request.get("user_id")
    user_query = request.get("user_query")

    if not user_id or not user_query:
        raise HTTPException(status_code=400, detail="'user_id' and 'user_query' are required.")

    if user_id not in user_memories:
        raise HTTPException(
            status_code=404, 
            detail="User session not found. Please start a new conversation."
        )
    
    memory = user_memories[user_id]
    
    # Call the run_query function from the graph.py module
    output = run_query(rag_graph, retriever, user_id, user_query, memory)
    assistant_response = output.get("final_answer", "Sorry, something went wrong.")
    
    return {
        "message": assistant_response,
        "user_id": user_id
    }

# To run the application:
# Run the server from your terminal: `uvicorn app:app --reload`