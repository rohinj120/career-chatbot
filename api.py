from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio

from chatbot import run_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str


async def generate_stream(message: str):
    answer = run_pipeline(message)

    # Simulated token streaming
    words = answer.split()

    for word in words:
        yield word + " "
        await asyncio.sleep(0.03)


@app.post("/chat-stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        generate_stream(request.message),
        media_type="text/plain"
    )