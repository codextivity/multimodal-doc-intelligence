# app/api/routes/chat.py — full updated file

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

class ChatResponse(BaseModel):
    answer: str
    question: str
    route: str = ""

def parse_history(history: list[dict]) -> list:
    """Converts client history format to LangChain messages."""
    messages = []
    for item in history:
        if item["role"] == "human":
            messages.append(HumanMessage(content=item["content"]))
        elif item["role"] == "ai":
            messages.append(AIMessage(content=item["content"]))
    return messages

@router.post("", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    """
    Ask questions about ingested documents.

    The agent automatically routes to the best handler:
    - text:    questions about written document content
    - vision:  questions about charts, images, diagrams
    - tools:   questions requiring calculation
    - compare: questions comparing text and visual content
    - web:     questions requiring current web information

    Every answer includes source citations.
    """
    agent = request.app.state.agent

    if agent is None:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested. Use POST /ingest first."
        )

    if not body.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )

    chat_history = parse_history(body.history)

    from app.core.multimodal_agent import run_multimodal_agent
    answer = run_multimodal_agent(agent, body.message, chat_history)

    return ChatResponse(
        answer=answer,
        question=body.message
    )