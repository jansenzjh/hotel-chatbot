"""
Configuration for the chatbot application.
Loads environment variables and sets up constants.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Environment Variables ---
# --- Environment Variables ---
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "airbnb")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# --- Model Configuration ---
# --- Model Configuration ---
EMBEDDING_MODEL_NAME = 'nomic-embed-text'
VECTOR_DIMENSION = 768  # Vector dimension for nomic-embed-text
GENERATIVE_MODEL_NAME = 'gemini-2.5-flash-lite'

# --- RAG Parameters ---
MATCH_THRESHOLD = 0.5  # Similarity threshold
MATCH_COUNT = 5        # Number of documents to retrieve
