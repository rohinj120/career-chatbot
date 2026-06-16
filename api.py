import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from chatbot import run_pipeline

logger = logging.getLogger(__name__)

fastapi_app = FastAPI()
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)

_executor = ThreadPoolExecutor(max_workers=2)
_active_sessions: set[str] = set()


@sio.event
async def connect(sid, environ):
    _active_sessions.add(sid)
    logger.info("Client connected: %s", sid)


@sio.event
async def disconnect(sid):
    _active_sessions.discard(sid)
    logger.info("Client disconnected: %s", sid)


@sio.event
async def send_message(sid, data):
    message = (data or {}).get("message", "").strip()
    if not message:
        return

    await sio.emit("typing", {}, to=sid)

    try:
        loop = asyncio.get_event_loop()
        answer, sources = await loop.run_in_executor(_executor, run_pipeline, message)

        words = answer.split()
        for word in words:
            if sid not in _active_sessions:
                return
            await sio.emit("token", {"token": word + " "}, to=sid)
            await asyncio.sleep(0.03)

        await sio.emit("sources", {"sources": sources}, to=sid)
        await sio.emit("done", {}, to=sid)

    except Exception:
        logger.exception("Error processing message from %s", sid)
        await sio.emit("error", {"message": "Something went wrong. Please try again."}, to=sid)