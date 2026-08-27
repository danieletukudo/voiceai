import os

from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "static-retrieval-mrl-en-v1")
GEMINI_LIVE_MODEL = os.getenv(
    "GEMINI_LIVE_MODEL",
    "gemini-2.5-flash-native-audio-preview-12-2025",
)
GEMINI_LIVE_VOICE = os.getenv("GEMINI_LIVE_VOICE", "Puck")

COLLECTION_NAME = "documents"
VECTOR_SIZE = 1024

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
