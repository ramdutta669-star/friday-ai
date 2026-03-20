from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager

import uvicorn
import logging
import json
import time
import re
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor

# ❌ edge_tts remove (optional if problem hoy)
# import edge_tts

from models import ChatRequest, ChatResponse, TTSRequest
from vector_store import VectorStoreService
from groq_service import GroqService, AllGroqApisFailedError
from realtime_service import RealtimeGroqService
from chat_service import ChatService

from config import (
    VECTOR_STORE_DIR, GROQ_API_KEYS, GROQ_MODEL, TAVILY_API_KEY,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHAT_HISTORY_TURNS,
    ASSISTANT_NAME
)

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
)
logger = logging.getLogger("F.I.R.D.A.Y")

# =========================
# GLOBAL SERVICES
# =========================
vector_store_service = None
groq_service = None
realtime_service = None
chat_service = None

# =========================
# STARTUP (lifespan)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store_service, groq_service, realtime_service, chat_service

    try:
        logger.info("🚀 Starting FRIDAY...")

        vector_store_service = VectorStoreService()
        vector_store_service.create_vector_store()

        groq_service = GroqService(vector_store_service)
        realtime_service = RealtimeGroqService(vector_store_service)
        chat_service = ChatService(groq_service, realtime_service)

        logger.info("✅ All services initialized!")

        yield

    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

# =========================
# APP INIT
# =========================
app = FastAPI(
    title="F.R.I.D.A.Y API",
    lifespan=lifespan
)

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ROOT
# =========================
@app.get("/")
async def root():
    return {"message": "FRIDAY is running 🚀"}

# =========================
# HEALTH
# =========================
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "chat_service": chat_service is not None
    }

# =========================
# CHAT
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not chat_service:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        session_id = chat_service.get_or_create_session(request.session_id)
        response = chat_service.process_message(session_id, request.message)

        return ChatResponse(
            response=response,
            session_id=session_id
        )

    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# STREAM CHAT
# =========================
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):

    if not chat_service:
        raise HTTPException(status_code=503, detail="Service not ready")

    def generator():
        yield "data: Starting...\n\n"

        try:
            session_id = chat_service.get_or_create_session(request.session_id)
            for chunk in chat_service.process_message_stream(session_id, request.message):
                yield f"data: {chunk}\n\n"

            yield "data: DONE\n\n"

        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")

# =========================
# RUN (LOCAL ONLY)
# =========================
def run():
    uvicorn.run(
        "main:app",   # ✅ FIXED
        host="0.0.0.0",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    run()
