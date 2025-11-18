"""
Configuration for the chatbot application.
Loads environment variables and sets up constants.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Environment Variables ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# --- Model Configuration ---
EMBEDDING_MODEL_NAME = 'mxbai-embed-large'
VECTOR_DIMENSION = 1024  # Vector dimension for mxbai-embed-large
GENERATIVE_MODEL_NAME = 'gemini-2.5-flash-lite'

# --- RAG Parameters ---
MATCH_THRESHOLD = 0.5  # Similarity threshold
MATCH_COUNT = 5        # Number of documents to retrieve
