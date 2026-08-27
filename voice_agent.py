"""
LiveKit voice agent with document RAG as a tool call.

Uses Gemini Live native audio for speech in + LLM + speech out
(no separate STT/TTS). Retrieval stays LLM-free in lookup_documents.

Run:
  ./venv/bin/python voice_agent.py console   # local mic test
  ./venv/bin/python voice_agent.py start     # join LiveKit rooms
"""

import asyncio
import os
from typing import Any

from dotenv import load_dotenv

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
load_dotenv()

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import google
from qdrant_client import AsyncQdrantClient
from sentence_transformers import SentenceTransformer

from rag_config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    GEMINI_LIVE_MODEL,
    GEMINI_LIVE_VOICE,
    QDRANT_API_KEY,
    QDRANT_URL,
    TOP_K,
)

AGENT_INSTRUCTIONS = """You are a warm, professional customer care assistant speaking with a caller on the phone.

Tone:
- Sound natural and helpful, like a real support agent — never like you are reading from a file or script.
- Keep replies short and conversational for voice.
- Do not mention documents, files, pages, sources, databases, retrieval, or that you looked something up.
- Do not say things like "according to the document" or "based on what I found."

How to answer:
- For questions about the company, policies, people, products, services, or similar details, use the lookup_documents tool first.
- Use only information returned by the tool.
- If the tool has useful info, answer clearly in your own words as a customer care agent.
- If the tool has nothing useful, say something natural like: "I'm sorry, we don't have that information available right now." or "I don't have those details on hand." Offer to help with something else if appropriate.
- Never invent facts.
"""


async def retrieve_chunks(
    question: str,
    embedding_model: SentenceTransformer,
    qdrant_client: AsyncQdrantClient,
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    loop = asyncio.get_event_loop()
    query_vector = await loop.run_in_executor(
        None,
        lambda: embedding_model.encode([question])[0].tolist(),
    )

    search_result = await qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    hits = search_result if isinstance(search_result, list) else getattr(search_result, "points", [])
    return [hit.payload for hit in hits if hit.payload]


def format_chunks(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "No company information was found for this question."

    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "")
        blocks.append(f"Info {i}:\n{text}")
    return "\n\n".join(blocks)


class DocumentVoiceAgent(Agent):
    def __init__(
        self,
        embedding_model: SentenceTransformer,
        qdrant_client: AsyncQdrantClient,
    ) -> None:
        self._embedding_model = embedding_model
        self._qdrant_client = qdrant_client
        super().__init__(instructions=AGENT_INSTRUCTIONS)

    @function_tool
    async def lookup_documents(self, context: RunContext, query: str) -> str:
        """Look up company information needed to help the caller.

        Use this when the caller asks about company details, policies, people,
        products, services, or anything you need internal info for.
        Pass a clear search query based on what they asked.
        """
        chunks = await retrieve_chunks(query, self._embedding_model, self._qdrant_client)
        return format_chunks(chunks)

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Greet the caller briefly and warmly as a customer care assistant. Offer to help. Do not mention documents or files."
        )


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    _ = embedding_model.encode("warmup")

    qdrant_client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)

    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            model=GEMINI_LIVE_MODEL,
            voice=GEMINI_LIVE_VOICE,
        ),
    )

    await session.start(
        agent=DocumentVoiceAgent(embedding_model, qdrant_client),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
